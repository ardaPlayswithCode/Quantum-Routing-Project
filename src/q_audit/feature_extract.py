"""Normalise an input circuit and extract the features the audit needs.

The important boundary here is the **unroll**: everything downstream (VF2,
SABRE, our ESP model) assumes a circuit of 1- and 2-qubit gates.  A 3q+ gate
that leaks past this point does not raise -- it silently makes ``VF2Layout``
bail with ``MORE_THAN_2Q`` and the whole audit degrades to a no-op.  So we
assert instead of hoping.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.transpiler.preset_passmanagers import common

from .errors import UnsupportedCircuitError
from .models import CalibrationSnapshot

CONTROL_FLOW_OPS = frozenset({"if_else", "for_loop", "while_loop", "switch_case", "box"})
_STRUCTURAL_OPS = frozenset({"barrier", "measure", "reset", "delay"})


@dataclass(frozen=True)
class DagOpStats:
    """Operation counts that drive VF2's scoring exponents.

    Empirically (see the canary test ``test_vf2_diagonal_exponent_is_one_qubit_op_count``)
    VF2's average-error scorer computes::

        score = 1 - prod_q (1 - diag[q]) ** n1q(q) * prod_e (1 - edge[e]) ** n2q(e)

    where ``n1q(q)`` counts **every** 1-qubit DAG op on that wire -- including
    zero-duration ``rz`` frame changes and ``measure`` -- but not barriers, and
    2-qubit gates contribute only through the off-diagonal term.

    The diagonal is therefore a *per-1q-operation* rate, not a per-qubit total.
    A raw ``sx + readout + idle`` diagonal is mis-scaled: it multiplies the
    one-shot readout and idle penalties by however many 1q ops happen to land
    on that wire.
    """

    one_qubit_ops: int
    error_bearing_1q_ops: int
    measure_ops: int
    active_qubits: int

    @property
    def n_bar(self) -> float:
        """Mean 1q ops per active qubit -- the exponent VF2 will apply."""
        return self.one_qubit_ops / self.active_qubits if self.active_qubits else 0.0

    @property
    def g_bar(self) -> float:
        """Mean *error-bearing* 1q gates per active qubit."""
        return self.error_bearing_1q_ops / self.active_qubits if self.active_qubits else 0.0

    @property
    def m_bar(self) -> float:
        """Mean measurements per active qubit."""
        return self.measure_ops / self.active_qubits if self.active_qubits else 0.0

    def to_dict(self) -> dict:
        return {
            "one_qubit_ops": self.one_qubit_ops,
            "error_bearing_1q_ops": self.error_bearing_1q_ops,
            "measure_ops": self.measure_ops,
            "active_qubits": self.active_qubits,
            "n_bar": self.n_bar,
            "g_bar": self.g_bar,
            "m_bar": self.m_bar,
        }


_ERROR_BEARING_1Q = frozenset(
    {"sx", "x", "id", "sxdg", "y", "h", "u", "u3", "r", "rx", "ry"}
)


def dag_op_stats(circuit: QuantumCircuit) -> DagOpStats:
    """Count the ops VF2 will use as scoring exponents."""
    one_q = 0
    err_1q = 0
    measures = 0
    active: set[int] = set()
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
    """Everything the orchestrator needs to know about the input circuit."""

    name: str
    num_qubits: int
    num_clbits: int
    depth: int
    size: int
    two_qubit_gates: int
    two_qubit_depth: int
    interaction_edges: list[tuple[int, int]]
    interaction_degree_max: int
    t_est_s: float
    bound_parameters: dict[str, float] = field(default_factory=dict)
    has_control_flow: bool = False
    has_measurements: bool = False
    op_counts: dict[str, int] = field(default_factory=dict)
    op_stats: DagOpStats | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
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


def _stable_parameter_value(circuit_name: str, param: Parameter) -> float:
    """Deterministic pseudo-random binding in [0, 2*pi).

    Derived from a hash of (circuit name, parameter name) so the same circuit
    always gets the same values -- a re-run must be reproducible, and baseline
    and audited paths must see byte-identical inputs.
    """
    digest = hashlib.sha256(f"{circuit_name}:{param.name}".encode()).digest()
    frac = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return frac * 6.283185307179586


def bind_free_parameters(circuit: QuantumCircuit) -> tuple[QuantumCircuit, dict[str, float]]:
    """Bind unbound parameters to fixed pseudo-random values.

    Layout and routing are parameter-independent, but scheduling and ESP are
    not (a gate's duration can depend on its angle).  Binding deterministically
    keeps the audit reproducible; the report lists what was bound.
    """
    params = list(circuit.parameters)
    if not params:
        return circuit, {}
    values = {p.name: _stable_parameter_value(circuit.name, p) for p in params}
    bound = circuit.assign_parameters({p: values[p.name] for p in params})
    return bound, values


def detect_control_flow(circuit: QuantumCircuit) -> bool:
    return any(name in CONTROL_FLOW_OPS for name in circuit.count_ops())


def unroll_to_basis(circuit: QuantumCircuit, basis_gates: list[str]) -> QuantumCircuit:
    """Translate to the device gate set *without* applying any connectivity.

    ``target=None`` is deliberate: passing a target here drags in the coupling
    map and direction fix-ups, which are meaningless on a circuit that has not
    been laid out yet.
    """
    pm = common.generate_translation_passmanager(target=None, basis_gates=list(basis_gates))
    return pm.run(circuit)


def assert_at_most_two_qubit(circuit: QuantumCircuit) -> None:
    """Hard gate: nothing wider than 2 qubits may reach the layout passes."""
    offenders: dict[str, int] = {}
    for inst in circuit.data:
        name = inst.operation.name
        if name == "barrier":
            continue
        if len(inst.qubits) > 2:
            offenders[name] = offenders.get(name, 0) + 1
    if offenders:
        listing = ", ".join(f"{k} x{v}" for k, v in sorted(offenders.items()))
        raise UnsupportedCircuitError(
            f"Circuit still contains gates wider than 2 qubits after unrolling: {listing}.",
            hint="VF2Layout silently gives up on >2q gates, which would make the "
            "audit a no-op. Decompose these before auditing.",
        )


def interaction_graph(circuit: QuantumCircuit) -> list[tuple[int, int]]:
    """Undirected 2q interaction edges over *virtual* qubit indices."""
    edges: set[tuple[int, int]] = set()
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


def two_qubit_depth(circuit: QuantumCircuit) -> int:
    """Critical-path depth counting only 2q gates."""
    return circuit.depth(
        lambda inst: len(inst.qubits) == 2 and inst.operation.name not in _STRUCTURAL_OPS
    )


def extract_features(
    circuit: QuantumCircuit,
    snapshot: CalibrationSnapshot,
    *,
    reject_control_flow: bool = True,
) -> tuple[QuantumCircuit, QuantumCircuit, CircuitFeatures]:
    """Prepare a circuit for audit.

    Returns ``(bound_original, unrolled, features)``.  The *original* (bound but
    not unrolled) circuit is what gets fed to the preset pass managers, so both
    the baseline and audited paths see byte-identical input; the *unrolled* copy
    exists purely so VF2Layout sees a 1q/2q interaction graph.
    """
    notes: list[str] = []

    has_cf = detect_control_flow(circuit)
    if has_cf and reject_control_flow:
        raise UnsupportedCircuitError(
            "Circuit contains classical control flow "
            f"({sorted(set(circuit.count_ops()) & CONTROL_FLOW_OPS)}).",
            hint="The MVP's ESP and idle-time model assumes a single static "
            "schedule; a branchy circuit has no single duration. Pass "
            "--allow-control-flow to audit it anyway (numbers become an "
            "upper bound on the longest branch).",
        )
    if has_cf:
        notes.append(
            "Circuit contains control flow: duration and ESP are upper bounds over "
            "the statically-scheduled longest branch, not the expected value."
        )

    bound, bound_values = bind_free_parameters(circuit)
    if bound_values:
        notes.append(
            f"Bound {len(bound_values)} free parameter(s) to fixed pseudo-random "
            "values (seeded by circuit+parameter name) so the audit is reproducible."
        )

    unrolled = unroll_to_basis(bound, snapshot.basis_gates)
    assert_at_most_two_qubit(unrolled)

    edges = interaction_graph(unrolled)
    degree: dict[int, int] = {}
    for a, b in edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1

    two_q = sum(
        1
        for inst in unrolled.data
        if len(inst.qubits) == 2 and inst.operation.name not in _STRUCTURAL_OPS
    )
    tq_depth = two_qubit_depth(unrolled)
    median_2q = snapshot.median_two_qubit_duration()
    t_est = tq_depth * median_2q

    if t_est <= 0:
        notes.append(
            "Circuit has no 2-qubit gates: idle-time estimate is zero and the "
            "injected map reduces to static (1q + readout) error."
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


def connectivity_summary(features: CircuitFeatures, snapshot: CalibrationSnapshot) -> str:
    """One line describing how well the circuit's shape fits the device graph."""
    n = features.num_qubits
    if not features.interaction_edges:
        return "no 2-qubit interactions"
    density = len(features.interaction_edges) / max(1, n * (n - 1) / 2)
    device_density = len(snapshot.coupling) / max(1, snapshot.num_qubits * (snapshot.num_qubits - 1) / 2)
    shape = "dense" if density > 0.5 else "sparse"
    return (
        f"{len(features.interaction_edges)} interaction edges over {n} qubits "
        f"({shape}, density {density:.2f} vs device {device_density:.3f}), "
        f"max degree {features.interaction_degree_max}"
    )
