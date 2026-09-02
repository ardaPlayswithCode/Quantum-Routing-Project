"""clean up the input circuit and pull out the numbers the audit needs.

the important bit is the unroll. everything after this point (vf2, sabre, our
esp model) assumes 1q and 2q gates only. a 3q gate sneaking through doesn't
raise anything, it just makes VF2Layout give up with MORE_THAN_2Q and then the
whole audit is a no-op. so we check instead of hoping
"""

import hashlib
from dataclasses import dataclass, field

from qiskit.transpiler.preset_passmanagers import common

from .errors import UnsupportedCircuitError

CONTROL_FLOW_OPS = frozenset({"if_else", "for_loop", "while_loop", "switch_case", "box"})
_STRUCTURAL_OPS = frozenset({"barrier", "measure", "reset", "delay"})

_ERROR_BEARING_1Q = frozenset(
    {"sx", "x", "id", "sxdg", "y", "h", "u", "u3", "r", "rx", "ry"}
)


@dataclass(frozen=True)
class DagOpStats:
    """op counts that end up as vf2's scoring exponents.

    vf2 scores a layout as prod_q (1-diag[q])**n1q(q) * prod_e (1-edge[e])**n2q(e)
    where n1q counts every 1 qubit op on the wire, including rz frame changes
    and measure, but not barriers. so the diagonal is a per op rate and not a
    per qubit total. see test_vf2_diagonal_exponent_is_one_qubit_op_count, i
    had to check this empirically becuase it isn't documented anywhere
    """

    one_qubit_ops: int
    error_bearing_1q_ops: int
    measure_ops: int
    active_qubits: int

    @property
    def n_bar(self):
        """mean 1q ops per active qubit, ie the exponent vf2 uses"""
        return self.one_qubit_ops / self.active_qubits if self.active_qubits else 0.0

    @property
    def g_bar(self):
        """same but only gates that actually carry error"""
        return self.error_bearing_1q_ops / self.active_qubits if self.active_qubits else 0.0

    @property
    def m_bar(self):
        return self.measure_ops / self.active_qubits if self.active_qubits else 0.0

    def to_dict(self):
        return {
            "one_qubit_ops": self.one_qubit_ops,
            "error_bearing_1q_ops": self.error_bearing_1q_ops,
            "measure_ops": self.measure_ops,
            "active_qubits": self.active_qubits,
            "n_bar": self.n_bar,
            "g_bar": self.g_bar,
            "m_bar": self.m_bar,
        }


def dag_op_stats(circuit):
    one_q = 0
    err_1q = 0
    measures = 0
    active = set()
    for inst in circuit.data:
        name = inst.operation.name
        if name == "barrier":
            continue
        for q in inst.qubits:
            active.add(circuit.find_bit(q).index)
        if len(inst.qubits) != 1:
            continue
        one_q += 1
        if name == "measure":
            measures += 1
        elif name in _ERROR_BEARING_1Q:
            err_1q += 1
    return DagOpStats(
        one_qubit_ops=one_q,
        error_bearing_1q_ops=err_1q,
        measure_ops=measures,
        active_qubits=len(active) or circuit.num_qubits,
    )


@dataclass
class CircuitFeatures:
    """everything the audit needs to know about the input circuit"""

    name: str
    num_qubits: int
    num_clbits: int
    depth: int
    size: int
    two_qubit_gates: int
    two_qubit_depth: int
    interaction_edges: list
    interaction_degree_max: int
    t_est_s: float
    bound_parameters: dict = field(default_factory=dict)
    has_control_flow: bool = False
    has_measurements: bool = False
    op_counts: dict = field(default_factory=dict)
    op_stats: object = None
    notes: list = field(default_factory=list)

    def to_dict(self):
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "num_clbits": self.num_clbits,
            "depth": self.depth,
            "size": self.size,
            "two_qubit_gates": self.two_qubit_gates,
            "two_qubit_depth": self.two_qubit_depth,
            "interaction_edges": len(self.interaction_edges),
            "interaction_degree_max": self.interaction_degree_max,
            "t_est_s": self.t_est_s,
            "bound_parameters": self.bound_parameters,
            "has_control_flow": self.has_control_flow,
            "has_measurements": self.has_measurements,
            "op_counts": self.op_counts,
            "op_stats": self.op_stats.to_dict() if self.op_stats else None,
            "notes": self.notes,
        }


def _stable_parameter_value(circuit_name, param) -> float:
    # hash of (circuit name, param name) -> a fixed angle in [0, 2pi). needs to
    # be the same every run or the two paths aren't comparing the same circuit
    digest = hashlib.sha256(f"{circuit_name}:{param.name}".encode()).digest()
    frac = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return frac * 6.283185307179586


def bind_free_parameters(circuit):
    """bind any leftover parameters to fixed pseudo random angles.

    layout and routing don't care about angles but scheduling and esp do, since
    a gate duration can depend on its angle
    """
    params = list(circuit.parameters)
    if not params:
        return circuit, {}
    values = {p.name: _stable_parameter_value(circuit.name, p) for p in params}
    bound = circuit.assign_parameters({p: values[p.name] for p in params})
    return bound, values


def detect_control_flow(circuit):
    return any(name in CONTROL_FLOW_OPS for name in circuit.count_ops())


def unroll_to_basis(circuit, basis_gates):
    """translate to the device gate set without applying any connectivity.

    target=None on purpose. passing a target drags in the coupling map and the
    direction fixups, which mean nothing on a circuit with no layout yet
    """
    pm = common.generate_translation_passmanager(target=None, basis_gates=list(basis_gates))
    return pm.run(circuit)


def assert_at_most_two_qubit(circuit):
    """nothing wider than 2 qubits gets past here"""
    offenders = {}
    for inst in circuit.data:
        name = inst.operation.name
        if name == "barrier":
            continue
        if len(inst.qubits) > 2:
            offenders[name] = offenders.get(name, 0) + 1
    if offenders:
        listing = ", ".join(f"{k} x{v}" for k, v in sorted(offenders.items()))
        raise UnsupportedCircuitError(
            f"circuit still has gates wider than 2 qubits after unrolling: {listing}",
            hint="VF2Layout silently gives up on >2q gates so the audit would be a "
            "no-op. decompose these first.",
        )


def interaction_graph(circuit):
    """undirected 2q edges over virtual qubit indices"""
    edges = set()
    for inst in circuit.data:
        if inst.operation.name in _STRUCTURAL_OPS:
            continue
        if len(inst.qubits) != 2:
            continue
        a = circuit.find_bit(inst.qubits[0]).index
        b = circuit.find_bit(inst.qubits[1]).index
        if a != b:
            edges.add((min(a, b), max(a, b)))
    return sorted(edges)


def two_qubit_depth(circuit) -> int:
    """depth counting only the 2q gates"""
    return circuit.depth(
        lambda inst: len(inst.qubits) == 2 and inst.operation.name not in _STRUCTURAL_OPS
    )


def extract_features(circuit, snapshot, *, reject_control_flow=True):
    """get a circuit ready to audit.

    returns (bound, unrolled, features). the bound-but-not-unrolled one is what
    goes into the preset pass managers so both paths see identical input. the
    unrolled copy only exists so vf2 sees a 1q/2q interaction graph
    """
    notes = []

    has_cf = detect_control_flow(circuit)
    if has_cf and reject_control_flow:
        raise UnsupportedCircuitError(
            "circuit has classical control flow "
            f"({sorted(set(circuit.count_ops()) & CONTROL_FLOW_OPS)})",
            hint="our esp and idle model assumes one static schedule and a branchy "
            "circuit doesn't have one. pass --allow-control-flow to audit it anyway, "
            "the numbers become an upper bound on the longest branch.",
        )
    if has_cf:
        notes.append(
            "circuit has control flow, so duration and esp are an upper bound over the "
            "longest branch rather than an expected value"
        )

    bound, bound_values = bind_free_parameters(circuit)
    if bound_values:
        notes.append(
            f"bound {len(bound_values)} free parameter(s) to fixed pseudo-random values "
            "(seeded by circuit + parameter name) so re-runs match"
        )

    unrolled = unroll_to_basis(bound, snapshot.basis_gates)
    assert_at_most_two_qubit(unrolled)

    edges = interaction_graph(unrolled)
    degree = {}
    for a, b in edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1

    two_q = sum(
        1
        for inst in unrolled.data
        if len(inst.qubits) == 2 and inst.operation.name not in _STRUCTURAL_OPS
    )
    tq_depth = two_qubit_depth(unrolled)
    # rough guess at how long the circuit runs: 2q depth times the typical 2q
    # gate. not exact but we only need it to size the idle penalty
    median_2q = snapshot.median_two_qubit_duration()
    t_est = tq_depth * median_2q

    if t_est <= 0:
        notes.append(
            "circuit has no 2-qubit gates so the idle estimate is zero and the injected "
            "map is just 1q + readout error"
        )

    features = CircuitFeatures(
        name=circuit.name,
        num_qubits=circuit.num_qubits,
        num_clbits=circuit.num_clbits,
        depth=unrolled.depth(lambda i: i.operation.name != "barrier"),
        size=unrolled.size(lambda i: i.operation.name != "barrier"),
        two_qubit_gates=two_q,
        two_qubit_depth=tq_depth,
        interaction_edges=edges,
        interaction_degree_max=max(degree.values()) if degree else 0,
        t_est_s=t_est,
        bound_parameters=bound_values,
        has_control_flow=has_cf,
        has_measurements=any(i.operation.name == "measure" for i in circuit.data),
        op_counts={k: int(v) for k, v in unrolled.count_ops().items()},
        op_stats=dag_op_stats(unrolled),
        notes=notes,
    )
    return bound, unrolled, features


def connectivity_summary(features, snapshot):
    """one line on how well the circuit shape fits the device graph"""
    n = features.num_qubits
    if not features.interaction_edges:
        return "no 2-qubit interactions"
    density = len(features.interaction_edges) / max(1, n * (n - 1) / 2)
    device_pairs = max(1, snapshot.num_qubits * (snapshot.num_qubits - 1) / 2)
    device_density = len(snapshot.coupling) / device_pairs
    shape = "dense" if density > 0.5 else "sparse"
    return (
        f"{len(features.interaction_edges)} interaction edges over {n} qubits "
        f"({shape}, density {density:.2f} vs device {device_density:.3f}), "
        f"max degree {features.interaction_degree_max}"
    )
