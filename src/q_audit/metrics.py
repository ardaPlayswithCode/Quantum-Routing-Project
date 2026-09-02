"""measuring a transpiled circuit: esp, duration, swaps, how good the layout is"""

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict

from .physics import esp_from_terms, idle_infidelity
from .scheduling import schedule_circuit

# passes that add routing swaps. we watch the delta they cause instead of
# counting swap ops at the end, becuase translation decomposes every swap into
# native 2q gates and the count disappears
_ROUTING_PASSES = frozenset(
    {"SabreSwap", "SabreLayout", "BasicSwap", "LookaheadSwap", "StochasticSwap"}
)
# anything at or above this error isn't a noisy gate, it's a broken one.
# fake_sherbrooke ships 9 couplers with error == 1.0 and TrivialLayout (opt
# level 0 and 1) will happily route straight through one. esp is then exactly
# zero, ie the job is dead before it starts. finding that out before you spend
# queue time is basically the point of this tool
DEAD_INSTRUCTION_THRESHOLD = 0.5

# virtual z frame changes and friends, no duration and no error
_FREE_OPS = frozenset({"rz", "barrier", "delay", "z", "s", "sdg", "t", "tdg", "p", "u1"})


class MetricSet(BaseModel):
    """everything we measured about one transpiled circuit"""

    model_config = ConfigDict(frozen=True)

    label: str
    seed: int
    layout: list[int]
    depth: int
    size: int
    two_qubit_gates: int
    swap_gates: int
    duration_s: float
    esp: float
    esp_gate_only: float
    esp_idle_only: float
    total_idle_s: float
    worst_t2_on_layout_s: float | None
    worst_t1_on_layout_s: float | None
    mean_t2_on_layout_s: float | None
    worst_edge_error: float | None
    readout_error_sum: float
    # only set if somebody ran the aer harness
    hellinger_fidelity: float | None = None


@dataclass
class SwapCounter:
    """pass manager callback that records swaps added by routing"""

    added: int = 0
    _prev: int = 0
    _seen_routing: bool = False
    per_pass: dict = field(default_factory=dict)

    def __call__(self, **kwargs):
        pass_ = kwargs.get("pass_")
        dag = kwargs.get("dag")
        if pass_ is None or dag is None:
            return
        name = type(pass_).__name__
        try:
            count = dag.count_ops().get("swap", 0)
        except Exception:  # never let instrumentation break a transpile
            return
        if name in _ROUTING_PASSES:
            delta = max(0, count - self._prev)
            self.added += delta
            self._seen_routing = True
            if delta:
                self.per_pass[name] = self.per_pass.get(name, 0) + delta
        self._prev = count


def instruction_error(snapshot, name, qubits) -> float:
    """calibrated error for one instruction on specific physical qubits"""
    if name in _FREE_OPS:
        return 0.0
    if name == "measure":
        return snapshot.qubit(qubits[0]).readout_error or 0.0
    if name == "reset":
        return 0.0
    if len(qubits) == 1:
        # unknown 1q gates get the sx error too, best proxy we have
        return snapshot.qubit(qubits[0]).sx_error or 0.0
    if len(qubits) == 2:
        edge = snapshot.edge(qubits[0], qubits[1])
        if edge is None or edge.error is None:
            return 0.0
        return edge.error
    # 3q+ shouldn't survive to a routed circuit
    return 0.0


def gate_error_terms(circuit, snapshot):
    """per instruction error terms, plus a small breakdown for the report"""
    errors = []
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


def idle_error_terms(schedule, snapshot):
    """(duration, t1, t2) per idle window, ready for esp_from_terms"""
    out = []
    for qubit, duration in schedule.idle_windows():
        cal = snapshot.qubit(qubit)
        out.append((duration, cal.t1, cal.t2))
    return out


def unusable_instructions(circuit, snapshot, *, threshold=DEAD_INSTRUCTION_THRESHOLD):
    """instructions the calibration says just cannot work, deduplicated"""
    found = {}
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


def active_physical_qubits(circuit):
    """physical qubits with at least one real instruction on them"""
    active = set()
    for inst in circuit.data:
        if inst.operation.name == "barrier":
            continue
        for q in inst.qubits:
            active.add(circuit.find_bit(q).index)
    return sorted(active)


def layout_qubits(circuit):
    """where the original virtual qubits ended up, in virtual order"""
    layout = getattr(circuit, "layout", None)
    if layout is not None:
        try:
            return list(layout.initial_index_layout(filter_ancillas=True))
        except Exception:  # no layout attached, fall back to whatever is active
            pass
    return active_physical_qubits(circuit)


def compute_metrics(circuit, snapshot, target, *, label, seed, swaps, schedule_method="asap"):
    """measure one transpiled circuit. returns (metrics, schedule)"""
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


def worst_idle_penalty(schedule, snapshot, top=5):
    """the idle windows costing the most fidelity, ie the report's why column"""
    rows = []
    for qubit, duration in schedule.idle_windows():
        cal = snapshot.qubit(qubit)
        err = idle_infidelity(duration, cal.t1, cal.t2)
        rows.append(
            {
                "qubit": qubit,
                "idle_s": duration,
                "t1_s": cal.t1,
                "t2_s": cal.t2,
                "infidelity": err,
            }
        )
    rows.sort(key=lambda r: r["infidelity"], reverse=True)
    return rows[:top]
