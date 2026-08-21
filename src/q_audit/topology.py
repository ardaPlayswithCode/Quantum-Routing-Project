"""ASCII rendering of a device coupling graph.

Coordinates are *derived* from the coupling map, not hard-coded for Eagle: IBM
numbers qubits sequentially along each row, so the edges with consecutive
indices form the horizontal chains and everything else is a vertical connector.
BFS over the chain graph then pins each chain to a row and an x-offset.  This
works for heavy-hex (Eagle/Heron) and for the older lattice devices; anything it
cannot lay out falls back to a plain listing rather than drawing a lie.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import CalibrationSnapshot
from .physics import percentile

# Rendered cell is a 3-character right-aligned index plus a 1-character
# membership marker, then a 1-character link to the next column.
_CELL = 3
_MARK = {"both": "*", "audited": "a", "baseline": "b", "none": " "}


@dataclass(frozen=True)
class Grid:
    coords: dict[int, tuple[int, int]]  # qubit -> (row, col)
    ok: bool
    reason: str = ""

    @property
    def n_rows(self) -> int:
        return 1 + max((r for r, _ in self.coords.values()), default=0)

    @property
    def n_cols(self) -> int:
        return 1 + max((c for _, c in self.coords.values()), default=0)


def _chains(coupling: list[tuple[int, int]]) -> tuple[dict[int, int], dict[int, int], dict[int, list[int]]]:
    """Group qubits into horizontal chains of consecutive indices."""
    adj: dict[int, set[int]] = {}
    for a, b in coupling:
        if abs(a - b) == 1:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

    nodes = sorted({q for edge in coupling for q in edge})
    chain_of: dict[int, int] = {}
    members: dict[int, list[int]] = {}
    next_id = 0
    for node in nodes:
        if node in chain_of:
            continue
        stack = [node]
        group: list[int] = []
        chain_of[node] = next_id
        while stack:
            cur = stack.pop()
            group.append(cur)
            for nb in adj.get(cur, ()):
                if nb not in chain_of:
                    chain_of[nb] = next_id
                    stack.append(nb)
        members[next_id] = sorted(group)
        next_id += 1

    pos_in_chain = {q: i for group in members.values() for i, q in enumerate(group)}
    return chain_of, pos_in_chain, members


def build_grid(snapshot: CalibrationSnapshot) -> Grid:
    coupling = [tuple(e) for e in snapshot.coupling]  # type: ignore[misc]
    if not coupling:
        return Grid({}, ok=False, reason="no coupling map")

    chain_of, pos_in_chain, members = _chains(coupling)
    if not members:
        return Grid({}, ok=False, reason="no chains found")

    # Chain adjacency via the non-consecutive ("vertical") edges.
    links: dict[int, list[tuple[int, int]]] = {}
    for a, b in coupling:
        ca, cb = chain_of.get(a), chain_of.get(b)
        if ca is None or cb is None or ca == cb:
            continue
        links.setdefault(ca, []).append((a, b))
        links.setdefault(cb, []).append((b, a))

    start = min(members, key=lambda c: min(members[c]))
    row_of: dict[int, int] = {start: 0}
    offset_of: dict[int, int] = {start: 0}
    queue = [start]
    while queue:
        chain = queue.pop(0)
        for here, there in links.get(chain, ()):
            other = chain_of[there]
            if other in row_of:
                continue
            direction = 1 if min(members[other]) > min(members[chain]) else -1
            row_of[other] = row_of[chain] + direction
            offset_of[other] = (
                offset_of[chain] + pos_in_chain[here] - pos_in_chain[there]
            )
            queue.append(other)

    if len(row_of) != len(members):
        return Grid({}, ok=False, reason="coupling graph is not a single component")

    raw: dict[int, tuple[int, int]] = {}
    for chain, group in members.items():
        for q in group:
            raw[q] = (row_of[chain], offset_of[chain] + pos_in_chain[q])

    min_row = min(r for r, _ in raw.values())
    min_col = min(c for _, c in raw.values())
    coords = {q: (r - min_row, c - min_col) for q, (r, c) in raw.items()}

    n_cols = 1 + max(c for _, c in coords.values())
    if n_cols > 40:
        return Grid(coords, ok=False, reason=f"grid too wide to render ({n_cols} columns)")
    return Grid(coords, ok=True)


def t2_tiers(snapshot: CalibrationSnapshot) -> tuple[float, float]:
    """33rd/66th percentile T2 cutoffs used for the red/yellow/green colouring."""
    values = [q.t2 for q in snapshot.qubits if q.t2 is not None]
    if not values:
        return (0.0, 0.0)
    return (percentile(values, 33.3), percentile(values, 66.6))


def tier_of(t2: float | None, cuts: tuple[float, float]) -> str:
    if t2 is None:
        return "unknown"
    if t2 <= cuts[0]:
        return "low"
    if t2 <= cuts[1]:
        return "mid"
    return "high"


_STYLE = {
    "low": "red",
    "mid": "yellow",
    "high": "green",
    "unknown": "grey50",
}


def render_topology(
    snapshot: CalibrationSnapshot,
    *,
    baseline_layout: list[int],
    audited_layout: list[int],
    markup: bool = True,
) -> list[str]:
    """Render the device as ASCII, coloured by T2 tier, with both layouts marked.

    Returns a list of lines.  With ``markup=True`` the lines carry rich markup
    tags; with ``markup=False`` they are plain text (used for ``--no-color`` and
    for tests).
    """
    grid = build_grid(snapshot)
    cuts = t2_tiers(snapshot)
    base = set(baseline_layout)
    audit = set(audited_layout)

    if not grid.ok:
        return [
            f"(topology map unavailable: {grid.reason})",
            f"baseline layout: {sorted(base)}",
            f"audited  layout: {sorted(audit)}",
        ]

    n_rows, n_cols = grid.n_rows, grid.n_cols
    by_cell = {(r, c): q for q, (r, c) in grid.coords.items()}
    horizontal = {
        (min(a, b), max(a, b))
        for a, b in snapshot.coupling
        if a in grid.coords
        and b in grid.coords
        and grid.coords[a][0] == grid.coords[b][0]
    }
    vertical = {
        (min(a, b), max(a, b))
        for a, b in snapshot.coupling
        if a in grid.coords
        and b in grid.coords
        and grid.coords[a][0] != grid.coords[b][0]
    }
    vertical_cols: dict[int, set[int]] = {}
    for a, b in vertical:
        (ra, ca), (rb, _) = grid.coords[a], grid.coords[b]
        top = min(ra, rb)
        vertical_cols.setdefault(top, set()).add(ca)

    def cell(q: int) -> str:
        if q in base and q in audit:
            member = "both"
        elif q in audit:
            member = "audited"
        elif q in base:
            member = "baseline"
        else:
            member = "none"
        # The marker is ASCII on purpose: colour is stripped the moment the
        # output is piped, and a report that loses its meaning under `| less`
        # is not a report.
        text = f"{q:>{_CELL}d}{_MARK[member]}"
        if not markup:
            return text
        style = _STYLE[tier_of(snapshot.qubit(q).t2, cuts)]
        if member == "both":
            style = f"bold {style} on grey35"
        elif member == "audited":
            style = f"bold {style} on dark_green"
        elif member == "baseline":
            style = f"bold {style} on grey19"
        return f"[{style}]{text}[/]"

    lines: list[str] = []
    for r in range(n_rows):
        row_chars: list[str] = []
        for c in range(n_cols):
            q = by_cell.get((r, c))
            row_chars.append(cell(q) if q is not None else " " * (_CELL + 1))
            if c < n_cols - 1:
                right = by_cell.get((r, c + 1))
                linked = (
                    q is not None
                    and right is not None
                    and (min(q, right), max(q, right)) in horizontal
                )
                row_chars.append("─" if linked else " ")
        line = "".join(row_chars).rstrip()
        if line.strip():
            lines.append(line)
        if r in vertical_cols and r < n_rows - 1:
            conn: list[str] = []
            for c in range(n_cols):
                conn.append("  │ " if c in vertical_cols[r] else " " * (_CELL + 1))
                if c < n_cols - 1:
                    conn.append(" ")
            lines.append("".join(conn).rstrip())
    return lines


def legend_lines(snapshot: CalibrationSnapshot, *, markup: bool = True) -> list[str]:
    lo, hi = t2_tiers(snapshot)
    marks = "markers:  b = baseline only   a = audited only   * = both"
    if markup:
        return [
            f"T2 tiers: [red]red[/] <= {lo * 1e6:.0f}us   "
            f"[yellow]yellow[/] <= {hi * 1e6:.0f}us   "
            f"[green]green[/] > {hi * 1e6:.0f}us",
            marks,
        ]
    return [
        f"T2 tiers: red <= {lo * 1e6:.0f}us, yellow <= {hi * 1e6:.0f}us, "
        f"green > {hi * 1e6:.0f}us",
        marks,
    ]
