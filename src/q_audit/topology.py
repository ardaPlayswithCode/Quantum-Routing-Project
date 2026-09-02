"""ascii picture of the device coupling graph.

the coordinates aren't hard coded for eagle, they get worked out from the
coupling map. ibm numbers qubits along each row, so edges between consecutive
indices are the horizontal chains and everything else is a vertical connector.
then bfs over the chains to pin each one to a row and an x offset. works for
heavy hex and the older lattice devices. if it can't lay something out it
prints a plain list instead of drawing a lie
"""

from dataclasses import dataclass

from .physics import percentile

# each cell is a 3 char right aligned index plus a 1 char membership marker
_CELL = 3
_MARK = {"both": "*", "audited": "a", "baseline": "b", "none": " "}


@dataclass(frozen=True)
class Grid:
    coords: dict  # qubit -> (row, col)
    ok: bool
    reason: str = ""

    @property
    def n_rows(self):
        return 1 + max((r for r, _ in self.coords.values()), default=0)

    @property
    def n_cols(self):
        return 1 + max((c for _, c in self.coords.values()), default=0)


def _chains(coupling):
    """group qubits into horizontal chains of consecutive indices"""
    adj = {}
    for a, b in coupling:
        if abs(a - b) == 1:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

    nodes = sorted({q for edge in coupling for q in edge})
    chain_of = {}
    members = {}
    next_id = 0
    for node in nodes:
        if node in chain_of:
            continue
        stack = [node]
        group = []
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


def build_grid(snapshot):
    coupling = [tuple(e) for e in snapshot.coupling]
    if not coupling:
        return Grid({}, ok=False, reason="no coupling map")

    chain_of, pos_in_chain, members = _chains(coupling)
    if not members:
        return Grid({}, ok=False, reason="no chains found")

    # chains connect to each other through the non consecutive edges
    links = {}
    for a, b in coupling:
        ca = chain_of.get(a)
        cb = chain_of.get(b)
        if ca is None or cb is None or ca == cb:
            continue
        links.setdefault(ca, []).append((a, b))
        links.setdefault(cb, []).append((b, a))

    start = min(members, key=lambda c: min(members[c]))
    row_of = {start: 0}
    offset_of = {start: 0}
    queue = [start]
    while queue:
        chain = queue.pop(0)
        for here, there in links.get(chain, ()):
            other = chain_of[there]
            if other in row_of:
                continue
            direction = 1 if min(members[other]) > min(members[chain]) else -1
            row_of[other] = row_of[chain] + direction
            offset_of[other] = offset_of[chain] + pos_in_chain[here] - pos_in_chain[there]
            queue.append(other)

    if len(row_of) != len(members):
        return Grid({}, ok=False, reason="coupling graph is not a single component")

    raw = {}
    for chain, group in members.items():
        for q in group:
            raw[q] = (row_of[chain], offset_of[chain] + pos_in_chain[q])

    # shift everything so the top left is (0, 0)
    min_row = min(r for r, _ in raw.values())
    min_col = min(c for _, c in raw.values())
    coords = {q: (r - min_row, c - min_col) for q, (r, c) in raw.items()}

    n_cols = 1 + max(c for _, c in coords.values())
    if n_cols > 40:
        return Grid(coords, ok=False, reason=f"grid too wide to render ({n_cols} columns)")
    return Grid(coords, ok=True)


def t2_tiers(snapshot):
    """33rd/66th percentile T2, used for the red/yellow/green colours"""
    values = [q.t2 for q in snapshot.qubits if q.t2 is not None]
    if not values:
        return (0.0, 0.0)
    return (percentile(values, 33.3), percentile(values, 66.6))


def tier_of(t2, cuts):
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


def render_topology(snapshot, *, baseline_layout, audited_layout, markup=True):
    """draw the device, coloured by T2 tier, with both layouts marked.

    markup=False gives plain lines, for tests and for piping
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
        if a in grid.coords and b in grid.coords and grid.coords[a][0] == grid.coords[b][0]
    }
    vertical = {
        (min(a, b), max(a, b))
        for a, b in snapshot.coupling
        if a in grid.coords and b in grid.coords and grid.coords[a][0] != grid.coords[b][0]
    }
    vertical_cols = {}
    for a, b in vertical:
        (ra, ca), (rb, _) = grid.coords[a], grid.coords[b]
        top = min(ra, rb)
        vertical_cols.setdefault(top, set()).add(ca)

    def cell(q):
        if q in base and q in audit:
            member = "both"
        elif q in audit:
            member = "audited"
        elif q in base:
            member = "baseline"
        else:
            member = "none"
        # the marker is ascii on purpose, colour disappears the second you pipe
        # this anywhere and a map that means nothing under `| less` is useless
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

    lines = []
    for r in range(n_rows):
        row_chars = []
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
            conn = []
            for c in range(n_cols):
                conn.append("  │ " if c in vertical_cols[r] else " " * (_CELL + 1))
                if c < n_cols - 1:
                    conn.append(" ")
            lines.append("".join(conn).rstrip())
    return lines


def legend_lines(snapshot, *, markup=True):
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
