"""turn a routed circuit into per qubit busy/idle intervals.

only file that knows about qiskit's scheduling passes. hands physics.py plain
(duration, t1, t2) tuples.

the accounting rule: a qubit only decoheres once you've touched it. time before
a qubit's first gate is not idle time, the qubit is sitting in |0> where
amplitude damping has nothing to damp and there's no superposition to dephase.
so idle windows are the gaps *between* instructions on a wire, which does
include the tail before the measurement
"""

from dataclasses import dataclass

from qiskit.transpiler import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import (
    ALAPScheduleAnalysis,
    ASAPScheduleAnalysis,
    TimeUnitConversion,
)

from .errors import TranspileAuditError

# a delay IS idle time, don't count it as busy
_IDLE_OPS = frozenset({"delay"})
_NON_INSTRUCTIONS = frozenset({"barrier"})


@dataclass(frozen=True)
class QubitTimeline:
    """busy and idle intervals for one physical qubit, in seconds"""

    qubit: int
    busy: tuple
    idle: tuple

    @property
    def total_idle(self):
        return sum(stop - start for start, stop in self.idle)

    @property
    def first_touch(self):
        return self.busy[0][0] if self.busy else None

    @property
    def last_touch(self):
        return self.busy[-1][1] if self.busy else None


@dataclass(frozen=True)
class ScheduleResult:
    timelines: dict
    total_duration_s: float
    method: str
    # instructions the target had no duration for. we treat them as instant,
    # which makes the circuit look shorter than it is, so the caller shows them
    unknown_durations: tuple = ()

    def total_idle_s(self):
        return sum(tl.total_idle for tl in self.timelines.values())

    def idle_windows(self):
        """(physical qubit, duration) for every idle gap"""
        out = []
        for q, tl in self.timelines.items():
            for start, stop in tl.idle:
                out.append((q, stop - start))
        return out


def _instruction_duration_dt(durations, name, qargs, instruction, unknown):
    """duration in dt units, from the target not a hard coded gate list.

    a timed target already says 0 for barrier and for rz frame changes, so
    there's no need to special case them, and hard coding them would be wrong
    on a backend where they aren't free
    """
    if name in _NON_INSTRUCTIONS:
        return 0
    if name in _IDLE_OPS:
        # delay carries its own duration, TimeUnitConversion already made it dt
        try:
            return int(instruction.duration)
        except (TypeError, ValueError):
            return 0
    try:
        return int(durations.get(name, list(qargs), unit="dt"))
    except Exception:  # not in the target, note it and move on
        unknown.add(name)
        return 0


def schedule_circuit(circuit, target, *, method="asap"):
    """schedule the circuit and pull out per qubit timelines.

    asap is the default and the conservative choice, it front loads the gates
    and leaves the idle time sitting in front of the measurement, which is what
    the hardware does when nobody inserted explicit delays
    """
    method = method.lower()
    if method not in ("asap", "alap"):
        raise TranspileAuditError(f"unknown scheduling method {method!r}, use asap or alap")

    dt = getattr(target, "dt", None)
    if not dt:
        raise TranspileAuditError(
            "backend target has no dt so we can't compute durations",
            hint="idle time analysis needs a timed target",
        )

    analysis = ASAPScheduleAnalysis if method == "asap" else ALAPScheduleAnalysis
    pm = PassManager([TimeUnitConversion(target=target), analysis(target=target)])
    try:
        scheduled = pm.run(circuit)
    except TranspilerError as exc:
        # qiskit's message here is "Duration of ecr on qubits [53, 41] is not
        # found" which is useless untill you know direction matters. two real
        # causes: a non-ISA gate survived, or a 2q gate is on the uncalibrated
        # orientation of a coupler
        raise TranspileAuditError(
            f"can't schedule the circuit against {getattr(target, 'description', 'the target')}: {exc}",
            hint="every instruction has to be ISA-valid including 2q gate direction. "
            "if this came from a relocation then the GateDirection fixup didn't run.",
        ) from exc
    node_start_time = pm.property_set.get("node_start_time")
    if not node_start_time:
        raise TranspileAuditError(
            "scheduling gave us no node_start_time, qiskit api probably changed",
            hint="check ASAPScheduleAnalysis in this qiskit version",
        )

    durations = target.durations()
    unknown = set()
    per_qubit = {}
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

    timelines = {}
    for q, spans in per_qubit.items():
        spans.sort()
        # merge overlapping busy spans first, otherwise a 2q gate that overlaps
        # a 1q one on the same wire shows up as negative idle
        merged = []
        for start, stop in spans:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], stop)
            else:
                merged.append([start, stop])
        idle = []
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
