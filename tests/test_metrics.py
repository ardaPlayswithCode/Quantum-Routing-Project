"""ESP measurement, SWAP counting and layout extraction."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

from q_audit.metrics import (
    SwapCounter,
    active_physical_qubits,
    compute_metrics,
    gate_error_terms,
    instruction_error,
    layout_qubits,
    worst_idle_penalty,
)


@pytest.fixture(scope="module")
def transpiled(request):
    backend = request.getfixturevalue("fake_backend")
    qc = QuantumCircuit(5)
    qc.h(0)
    for i in range(4):
        qc.cx(i, i + 1)
    qc.measure_all()
    counter = SwapCounter()
    pm = generate_preset_pass_manager(
        optimization_level=3, backend=backend, seed_transpiler=42
    )
    return pm.run(qc, callback=counter), counter


def test_instruction_error_lookup(snapshot):
    q = snapshot.qubits[0]
    assert instruction_error(snapshot, "sx", (0,)) == q.sx_error
    assert instruction_error(snapshot, "x", (0,)) == q.sx_error
    assert instruction_error(snapshot, "measure", (0,)) == q.readout_error
    assert instruction_error(snapshot, "rz", (0,)) == 0.0
    assert instruction_error(snapshot, "barrier", (0,)) == 0.0
    edge = snapshot.edges[0]
    assert instruction_error(snapshot, edge.gate, (edge.control, edge.target)) == edge.error
    # Reversed direction resolves to the same coupler.
    assert instruction_error(snapshot, edge.gate, (edge.target, edge.control)) == edge.error


def test_gate_error_breakdown_is_complete(transpiled, snapshot):
    circuit, _ = transpiled
    errors, breakdown = gate_error_terms(circuit, snapshot)
    assert errors
    assert all(0.0 <= e <= 1.0 for e in errors)
    assert breakdown["readout"] > 0
    assert breakdown["two_qubit"] > 0
    assert breakdown["one_qubit"] > 0


def test_swap_counter_sees_routing_swaps(transpiled):
    _, counter = transpiled
    assert counter.added >= 0
    # A 5-qubit line embeds perfectly on heavy-hex, so nothing should be added.
    assert counter.added == 0


def test_swap_counter_counts_when_routing_is_forced(fake_backend):
    """A fully-connected interaction graph cannot embed; SABRE must add SWAPs."""
    qc = QuantumCircuit(6)
    for i in range(6):
        for j in range(i + 1, 6):
            qc.cx(i, j)
    counter = SwapCounter()
    pm = generate_preset_pass_manager(
        optimization_level=3, backend=fake_backend, seed_transpiler=5
    )
    pm.run(qc, callback=counter)
    assert counter.added > 0


def test_layout_and_active_qubits_agree(transpiled):
    circuit, _ = transpiled
    layout = layout_qubits(circuit)
    assert len(layout) == 5
    assert set(layout) <= set(active_physical_qubits(circuit))


def test_compute_metrics_is_self_consistent(transpiled, snapshot, target):
    circuit, counter = transpiled
    metrics, schedule = compute_metrics(
        circuit, snapshot, target, label="probe", seed=42, swaps=counter.added
    )
    assert 0.0 < metrics.esp <= 1.0
    assert metrics.esp <= metrics.esp_gate_only
    assert metrics.esp <= metrics.esp_idle_only
    # ESP factorises exactly into its two halves.
    assert metrics.esp == pytest.approx(
        metrics.esp_gate_only * metrics.esp_idle_only, rel=1e-9
    )
    assert metrics.duration_s > 0
    assert metrics.total_idle_s == pytest.approx(schedule.total_idle_s())
    assert metrics.worst_t2_on_layout_s <= metrics.mean_t2_on_layout_s
    assert metrics.two_qubit_gates > 0
    assert metrics.depth > 0


def test_worst_idle_penalty_is_sorted(transpiled, snapshot, target):
    circuit, counter = transpiled
    _, schedule = compute_metrics(
        circuit, snapshot, target, label="probe", seed=42, swaps=counter.added
    )
    rows = worst_idle_penalty(schedule, snapshot, top=5)
    assert rows
    assert rows == sorted(rows, key=lambda r: r["infidelity"], reverse=True)


def test_dead_couplers_are_detected(snapshot, fake_backend):
    """fake_sherbrooke ships couplers with error == 1.0 and TrivialLayout
    (optimization_level 0/1) routes a 7-qubit line straight through one."""
    from qiskit import QuantumCircuit
    from qiskit.transpiler import generate_preset_pass_manager

    from q_audit.metrics import unusable_instructions

    dead_edges = [e for e in snapshot.edges if e.error is not None and e.error >= 0.999]
    assert dead_edges, "expected fake_sherbrooke to contain unusable couplers"

    qc = QuantumCircuit(7)
    qc.h(0)
    for i in range(6):
        qc.cx(i, i + 1)
    qc.measure_all()
    trivial = generate_preset_pass_manager(
        optimization_level=1, backend=fake_backend, seed_transpiler=42
    ).run(qc)
    found = unusable_instructions(trivial, snapshot)
    assert found, "a dead coupler on the trivial layout went unreported"
    assert found[0]["error"] >= 0.5


def test_zero_esp_when_a_gate_is_dead(snapshot, fake_backend, target):
    from qiskit import QuantumCircuit
    from qiskit.transpiler import generate_preset_pass_manager

    qc = QuantumCircuit(7)
    qc.h(0)
    for i in range(6):
        qc.cx(i, i + 1)
    qc.measure_all()
    trivial = generate_preset_pass_manager(
        optimization_level=1, backend=fake_backend, seed_transpiler=42
    ).run(qc)
    metrics, _ = compute_metrics(
        trivial, snapshot, target, label="trivial", seed=42, swaps=0
    )
    assert metrics.esp == 0.0
    assert metrics.worst_edge_error == 1.0


def test_healthy_circuit_reports_no_dead_hardware(transpiled, snapshot):
    from q_audit.metrics import unusable_instructions

    circuit, _ = transpiled
    assert unusable_instructions(circuit, snapshot) == []
