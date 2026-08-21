"""Measure a transpiled circuit: ESP, duration, routing overhead, layout quality."""

from __future__ import annotations

from dataclasses import dataclass, field

from qiskit import QuantumCircuit

from .models import CalibrationSnapshot, MetricSet
from .physics import esp_from_terms, idle_infidelity
from .scheduling import ScheduleResult, schedule_circuit

# Passes that introduce routing SWAPs.  We measure the delta they cause rather
# than counting `swap` ops in the final circuit, because the translation stage
# decomposes every SWAP into native 2q gates and the count vanishes.
_ROUTING_PASSES = frozenset({"SabreSwap", "SabreLayout", "BasicSwap", "LookaheadSwap", "StochasticSwap"})
# An instruction at or above this calibrated error is not a noisy gate, it is a
# broken one. fake_sherbrooke ships 9 couplers with error == 1.0, and Qiskit's
# TrivialLayout (optimization_level 0 and 1) will happily route a circuit
# straight through one -- ESP is then exactly zero and the job is dead on
# arrival. Finding that before you spend queue time is the whole point.
DEAD_INSTRUCTION_THRESHOLD = 0.5

# Gates whose reported error we take from the per-qubit 1q calibration.
_ONE_QUBIT_CALIBRATED = frozenset({"sx", "x", "id", "sxdg", "y", "h"})
# Virtual-Z style frame changes: no duration, no error.
_FREE_OPS = frozenset({"rz", "barrier", "delay", "z", "s", "sdg", "t", "tdg", "p", "u1"})


@dataclass
class SwapCounter:
    """Pass-manager callback that records SWAPs added by the routing stage."""

    added: int = 0
    _prev: int = 0
    _seen_routing: bool = False
    per_pass: dict[str, int] = field(default_factory=dict)

    def __call__(self, **kwargs) -> None:
        pass_ = kwargs.get("pass_")
        dag = kwargs.get("dag")
        if pass_ is None or dag is None:
            return
        name = type(pass_).__name__
        try:
            count = dag.count_ops().get("swap", 0)
        except Exception:  # noqa: BLE001 - never let instrumentation break a transpile
            return
        if name in _ROUTING_PASSES:
            delta = max(0, count - self._prev)
            self.added += delta
            self._seen_routing = True
            if delta:
                self.per_pass[name] = self.per_pass.get(name, 0) + delta
        self._prev = count


def instruction_error(
    snapshot: CalibrationSnapshot, name: str, qubits: tuple[int, ...]
) -> float:
    """Calibrated error for one instruction on specific *physical* qubits."""
    if name in _FREE_OPS:
        return 0.0
    if name == "measure":
        return snapshot.qubit(qubits[0]).readout_error or 0.0
    if name == "reset":
        return 0.0
    if len(qubits) == 1:
        q = snapshot.qubit(qubits[0])
        if name in _ONE_QUBIT_CALIBRATED:
            return q.sx_error or 0.0
        # Unknown 1q gate: the 1q calibration is the best proxy we have.
        return q.sx_error or 0.0
    if len(qubits) == 2:
        edge = snapshot.edge(qubits[0], qubits[1])
        return (edge.error if edge and edge.error is not None else 0.0)
    # >2q gates should never survive to a routed ISA circuit.
    return 0.0


def gate_error_terms(
    circuit: QuantumCircuit, snapshot: CalibrationSnapshot
) -> tuple[list[float], dict[str, float]]:
    """Per-instruction error terms plus a small breakdown for the report."""
    errors: list[float] = []
    breakdown = {"one_qubit": 0.0, "two_qubit": 0.0, "readout": 0.0}
    for inst in circuit.data:
        name = inst.operation.name
        if name in _FREE_OPS:
            continue
        qubits = tuple(circuit.find_bit(q).index for q in inst.qubits)
        err = instruction_error(snapshot, name, qubits)
        if err <= 0.0:
            continue
        errors.append(err)
        if name == "measure":
            breakdown["readout"] += err
        elif len(qubits) == 2:
            breakdown["two_qubit"] += err
        else:
            breakdown["one_qubit"] += err
    return errors, breakdown


def idle_error_terms(
    schedule: ScheduleResult, snapshot: CalibrationSnapshot
) -> list[tuple[float, float | None, float | None]]:
    """``(duration, t1, t2)`` for every idle window, ready for ``esp_from_terms``."""
    out: list[tuple[float, float | None, float | None]] = []
    for qubit, duration in schedule.idle_windows():
        cal = snapshot.qubit(qubit)
        out.append((duration, cal.t1, cal.t2))
    return out


def unusable_instructions(
    circuit: QuantumCircuit,
    snapshot: CalibrationSnapshot,
    *,
    threshold: float = DEAD_INSTRUCTION_THRESHOLD,
) -> list[dict]:
    """Instructions the calibration says cannot work, deduplicated."""
    found: dict[tuple[str, tuple[int, ...]], float] = {}
    for inst in circuit.data:
        name = inst.operation.name
        if name in _FREE_OPS:
            continue
        qubits = tuple(circuit.find_bit(q).index for q in inst.qubits)
        error = instruction_error(snapshot, name, qubits)
        if error >= threshold:
            found[(name, qubits)] = error
    return [
        {"instruction": name, "qubits": list(qubits), "error": error}
        for (name, qubits), error in sorted(found.items(), key=lambda kv: -kv[1])
    ]


def active_physical_qubits(circuit: QuantumCircuit) -> list[int]:
    """Physical qubits that carry at least one non-trivial instruction."""
    active: set[int] = set()
    for inst in circuit.data:
        if inst.operation.name == "barrier":
            continue
        for q in inst.qubits:
            active.add(circuit.find_bit(q).index)
    return sorted(active)


def layout_qubits(circuit: QuantumCircuit) -> list[int]:
    """Physical qubits holding the original virtual qubits, in virtual order."""
    layout = getattr(circuit, "layout", None)
    if layout is not None:
        try:
            return list(layout.initial_index_layout(filter_ancillas=True))
        except Exception:  # noqa: BLE001 - fall through to the active-qubit heuristic
            pass
    return active_physical_qubits(circuit)


def compute_metrics(
    circuit: QuantumCircuit,
    snapshot: CalibrationSnapshot,
    target,
    *,
    label: str,
    seed: int,
    swaps: int,
    schedule_method: str = "asap",
) -> tuple[MetricSet, ScheduleResult]:
    """Full measurement of one transpiled circuit."""
    schedule = schedule_circuit(circuit, target, method=schedule_method)
    gate_errors, _breakdown = gate_error_terms(circuit, snapshot)
    idle_terms = idle_error_terms(schedule, snapshot)

    esp = esp_from_terms(gate_errors, idle_terms)
    esp_gate_only = esp_from_terms(gate_errors, [])
    esp_idle_only = esp_from_terms([], idle_terms)

    phys = layout_qubits(circuit)
    active = active_physical_qubits(circuit)
    considered = phys or active

    t2s = [snapshot.qubit(q).t2 for q in considered if snapshot.qubit(q).t2 is not None]
    t1s = [snapshot.qubit(q).t1 for q in considered if snapshot.qubit(q).t1 is not None]
    readout_sum = sum(snapshot.qubit(q).readout_error or 0.0 for q in considered)

    worst_edge = None
    for inst in circuit.data:
        if len(inst.qubits) != 2 or inst.operation.name == "barrier":
            continue
        a = circuit.find_bit(inst.qubits[0]).index
        b = circuit.find_bit(inst.qubits[1]).index
        edge = snapshot.edge(a, b)
        if edge is None or edge.error is None:
            continue
        worst_edge = edge.error if worst_edge is None else max(worst_edge, edge.error)

    two_q = sum(
        1
        for inst in circuit.data
        if len(inst.qubits) == 2 and inst.operation.name != "barrier"
    )

    metrics = MetricSet(
        label=label,
        seed=seed,
        layout=phys,
        depth=circuit.depth(lambda i: i.operation.name != "barrier"),
        size=circuit.size(lambda i: i.operation.name != "barrier"),
        two_qubit_gates=two_q,
        swap_gates=swaps,
        duration_s=schedule.total_duration_s,
        esp=esp,
        esp_gate_only=esp_gate_only,
        esp_idle_only=esp_idle_only,
        total_idle_s=schedule.total_idle_s(),
        worst_t2_on_layout_s=min(t2s) if t2s else None,
        worst_t1_on_layout_s=min(t1s) if t1s else None,
        mean_t2_on_layout_s=(sum(t2s) / len(t2s)) if t2s else None,
        worst_edge_error=worst_edge,
        readout_error_sum=readout_sum,
    )
    return metrics, schedule


def worst_idle_penalty(
    schedule: ScheduleResult, snapshot: CalibrationSnapshot, top: int = 5
) -> list[dict]:
    """The idle windows that cost the most fidelity -- the report's 'why' column."""
    rows: list[dict] = []
    for qubit, duration in schedule.idle_windows():
        cal = snapshot.qubit(qubit)
        eps = idle_infidelity(duration, cal.t1, cal.t2)
        rows.append(
            {
                "qubit": qubit,
                "idle_s": duration,
                "t1_s": cal.t1,
                "t2_s": cal.t2,
                "infidelity": eps,
            }
        )
    rows.sort(key=lambda r: r["infidelity"], reverse=True)
    return rows[:top]
