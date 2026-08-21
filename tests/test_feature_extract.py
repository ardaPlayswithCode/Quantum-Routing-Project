"""Unrolling, the <=2q gate, parameter binding, control flow and t_est."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from q_audit.errors import UnsupportedCircuitError
from q_audit.feature_extract import (
    assert_at_most_two_qubit,
    bind_free_parameters,
    dag_op_stats,
    detect_control_flow,
    extract_features,
    interaction_graph,
    two_qubit_depth,
    unroll_to_basis,
)


def test_unroll_produces_only_basis_gates(snapshot, ghz7):
    unrolled = unroll_to_basis(ghz7, snapshot.basis_gates)
    allowed = set(snapshot.basis_gates) | {"barrier"}
    assert set(unrolled.count_ops()) <= allowed


def test_unroll_leaves_nothing_wider_than_two_qubits(snapshot):
    qc = QuantumCircuit(4)
    qc.ccx(0, 1, 2)
    qc.mcx([0, 1, 2], 3)
    unrolled = unroll_to_basis(qc, snapshot.basis_gates)
    assert_at_most_two_qubit(unrolled)  # must not raise


def test_assert_at_most_two_qubit_fires_on_wide_gates():
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    with pytest.raises(UnsupportedCircuitError) as exc:
        assert_at_most_two_qubit(qc)
    assert "wider than 2 qubits" in exc.value.message
    assert "VF2Layout silently gives up" in (exc.value.hint or "")


def test_assert_ignores_wide_barriers():
    qc = QuantumCircuit(5)
    qc.barrier()
    assert_at_most_two_qubit(qc)  # must not raise


def test_parameter_binding_is_deterministic():
    def make():
        qc = QuantumCircuit(2, name="param_probe")
        qc.rx(Parameter("theta"), 0)
        qc.ry(Parameter("phi"), 1)
        return qc

    _, first = bind_free_parameters(make())
    _, second = bind_free_parameters(make())
    assert first == second
    assert set(first) == {"theta", "phi"}
    assert all(0.0 <= v < 6.284 for v in first.values())


def test_parameter_binding_leaves_bound_circuits_alone(ghz7):
    bound, values = bind_free_parameters(ghz7)
    assert values == {}
    assert bound is ghz7


def test_control_flow_detection_and_rejection(snapshot):
    qc = QuantumCircuit(2, 1)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(1)
    assert detect_control_flow(qc)
    with pytest.raises(UnsupportedCircuitError):
        extract_features(qc, snapshot, reject_control_flow=True)
    _, _, features = extract_features(qc, snapshot, reject_control_flow=False)
    assert features.has_control_flow
    assert any("upper bound" in note for note in features.notes)


def test_interaction_graph_is_undirected_and_deduplicated():
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.cx(1, 2)
    assert interaction_graph(qc) == [(0, 1), (1, 2)]


def test_two_qubit_depth_counts_only_two_qubit_gates():
    qc = QuantumCircuit(2)
    for _ in range(10):
        qc.sx(0)
    qc.cx(0, 1)
    qc.cx(0, 1)
    assert two_qubit_depth(qc) == 2


def test_t_est_is_two_qubit_depth_times_median_duration(snapshot, ghz7):
    _, unrolled, features = extract_features(ghz7, snapshot)
    expected = two_qubit_depth(unrolled) * snapshot.median_two_qubit_duration()
    assert features.t_est_s == pytest.approx(expected)
    assert features.t_est_s > 0


def test_features_report_shape(snapshot, ghz7):
    _, _, features = extract_features(ghz7, snapshot)
    assert features.num_qubits == 7
    assert features.two_qubit_gates == 6
    assert features.interaction_degree_max == 2
    assert features.has_measurements
    assert features.to_dict()["op_stats"]["active_qubits"] == 7


def test_circuit_with_no_two_qubit_gates_notes_zero_t_est(snapshot):
    qc = QuantumCircuit(3)
    qc.h(range(3))
    qc.measure_all()
    _, _, features = extract_features(qc, snapshot)
    assert features.t_est_s == 0.0
    assert any("no 2-qubit gates" in note for note in features.notes)


def test_dag_op_stats_counts():
    qc = QuantumCircuit(2, 2)
    qc.sx(0)
    qc.sx(1)
    qc.rz(0.1, 0)
    qc.barrier()
    qc.ecr(0, 1)
    qc.measure(0, 0)
    stats = dag_op_stats(qc)
    assert stats.one_qubit_ops == 4  # 2 sx + 1 rz + 1 measure
    assert stats.error_bearing_1q_ops == 2
    assert stats.measure_ops == 1
    assert stats.active_qubits == 2
    assert stats.n_bar == 2.0
