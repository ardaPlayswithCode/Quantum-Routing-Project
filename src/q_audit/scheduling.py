"""Turn a routed circuit into per-qubit busy/idle intervals.

This is the only place that knows about Qiskit's scheduling passes.  It hands
``physics.py`` plain ``(duration_seconds, t1, t2)`` triples.

Idle accounting rule
--------------------
A qubit only decoheres once it has been touched.  Time before a qubit's first
instruction is *not* idle time: the qubit sits in |0>, where amplitude damping
has nothing to damp and dephasing has no superposition to dephase.  So idle
windows are exactly the gaps *between* consecutive instructions on a wire --
which naturally includes the tail between the last gate and the measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import (
    ALAPScheduleAnalysis,
    ASAPScheduleAnalysis,
    TimeUnitConversion,
)

from .errors import TranspileAuditError

# A Delay *is* idle time; never count it as busy.
_IDLE_OPS = frozenset({"delay"})
# Markers that are not instructions at all.
_NON_INSTRUCTIONS = frozenset({"barrier"})


@dataclass(frozen=True)
class QubitTimeline:
    """Busy and idle intervals for one physical qubit, in seconds."""

    qubit: int
    busy: tuple[tuple[float, float], ...]
    idle: tuple[tuple[float, float], ...]

    @property
    def total_idle(self) -> float:
        return sum(stop - start for start, stop in self.idle)

    @property
    def first_touch(self) -> float | None:
        return self.busy[0][0] if self.busy else None

    @property
    def last_touch(self) -> float | None:
        return self.busy[-1][1] if self.busy else None


@dataclass(frozen=True)
class ScheduleResult:
    timelines: dict[int, QubitTimeline]
    total_duration_s: float
    method: str
    # Instructions the target had no duration for. These are treated as
    # instantaneous, which *understates* the circuit's duration -- so the caller
    # surfaces them rather than letting the number quietly drift.
    unknown_durations: tuple[str, ...] = ()

    def total_idle_s(self) -> float:
        return sum(tl.total_idle for tl in self.timelines.values())

    def idle_windows(self) -> list[tuple[int, float]]:
        """``(physical_qubit, duration_seconds)`` for every idle gap."""
        out: list[tuple[int, float]] = []
        for q, tl in self.timelines.items():
            for start, stop in tl.idle:
                out.append((q, stop - start))
        return out


def _instruction_duration_dt(
    durations, name: str, qargs: tuple[int, ...], instruction, unknown: set[str]
) -> int:
    """Duration in dt, taken from the target rather than a hard-coded gate list.

    A timed target already reports 0 for ``barrier`` and for virtual-Z style
    frame changes such as ``rz``, so there is no need to special-case them here
    -- and hard-coding them would be wrong on a backend where they are not free.
    """
    if name in _NON_INSTRUCTIONS:
        return 0
    if name in _IDLE_OPS:
        # Delay carries its own duration; TimeUnitConversion has normalised it to dt.
        try:
            return int(instruction.duration)
        except (TypeError, ValueError):
            return 0
    try:
        return int(durations.get(name, list(qargs), unit="dt"))
    except Exception:  # noqa: BLE001 - not in the target; record and move on
        unknown.add(name)
        return 0


def schedule_circuit(
    circuit: QuantumCircuit,
    target,
    *,
    method: str = "asap",
) -> ScheduleResult:
    """Schedule ``circuit`` against ``target`` and extract per-qubit timelines.

    ``method`` is ``"asap"`` (default) or ``"alap"``.  ASAP is the conservative
    choice: it front-loads gates and leaves the accumulated idle time sitting in
    front of the measurement, which is what hardware does when the compiler has
    not inserted explicit delays.
    """
    method = method.lower()
    if method not in ("asap", "alap"):
        raise TranspileAuditError(f"Unknown scheduling method {method!r}; use asap or alap.")

    dt = getattr(target, "dt", None)
    if not dt:
        raise TranspileAuditError(
            "Backend target has no dt; cannot compute durations.",
            hint="Idle-time analysis requires a timed target.",
        )

    analysis = ASAPScheduleAnalysis if method == "asap" else ALAPScheduleAnalysis
    pm = PassManager([TimeUnitConversion(target=target), analysis(target=target)])
    try:
        scheduled = pm.run(circuit)
    except TranspilerError as exc:
        # Qiskit's scheduler refuses to time an instruction the target does not
        # know, and its message ("Duration of ecr on qubits [53, 41] is not
        # found") is opaque unless you know that direction matters. Two real
        # causes: a non-ISA gate survived, or a 2q gate sits on the
        # uncalibrated orientation of a coupler.
        raise TranspileAuditError(
            f"Cannot schedule the circuit against {getattr(target, 'description', 'the target')}: {exc}",
            hint="Every instruction must be ISA-valid, including 2q gate "
            "direction. If this came from a relocation, the GateDirection "
            "fix-up did not run.",
        ) from exc
    node_start_time = pm.property_set.get("node_start_time")
    if not node_start_time:
        raise TranspileAuditError(
            "Scheduling produced no node_start_time; Qiskit scheduling API drift.",
            hint="Check ASAPScheduleAnalysis in this Qiskit version.",
        )

    durations = target.durations()
    unknown: set[str] = set()
    per_qubit: dict[int, list[tuple[int, int]]] = {}
    horizon = 0
    for node, start in node_start_time.items():
        name = node.op.name
        qargs = tuple(scheduled.find_bit(q).index for q in node.qargs)
        dur = _instruction_duration_dt(durations, name, qargs, node.op, unknown)
        horizon = max(horizon, int(start) + dur)
        if dur <= 0 or name in _IDLE_OPS:
            continue
        for q in qargs:
            per_qubit.setdefault(q, []).append((int(start), int(start) + dur))

    timelines: dict[int, QubitTimeline] = {}
    for q, spans in per_qubit.items():
        spans.sort()
        merged: list[list[int]] = []
        for start, stop in spans:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], stop)
            else:
                merged.append([start, stop])
        idle: list[tuple[float, float]] = []
        for prev, nxt in zip(merged, merged[1:]):
            if nxt[0] > prev[1]:
                idle.append((prev[1] * dt, nxt[0] * dt))
        timelines[q] = QubitTimeline(
            qubit=q,
            busy=tuple((a * dt, b * dt) for a, b in merged),
            idle=tuple(idle),
        )

    return ScheduleResult(
        timelines=timelines,
        total_duration_s=horizon * dt,
        method=method,
        unknown_durations=tuple(sorted(unknown)),
    )
