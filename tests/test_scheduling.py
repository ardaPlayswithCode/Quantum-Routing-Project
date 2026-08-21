"""Idle-window extraction. The accounting rule is the thing under test."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

from q_audit.errors import TranspileAuditError
from q_audit.scheduling import schedule_circuit


@pytest.fixture(scope="module")
def routed(request):
    backend = request.getfixturevalue("fake_backend")
    qc = QuantumCircuit(4)
    qc.h(0)
    for i in range(3):
        qc.cx(i, i + 1)
    qc.measure_all()
    pm = generate_preset_pass_manager(
        optimization_level=3, backend=backend, seed_transpiler=42
    )
    return pm.run(qc)


@pytest.mark.parametrize("method", ["asap", "alap"])
def test_total_duration_matches_qiskit(routed, target, method):
    """Our horizon must agree with Qiskit's own estimate exactly."""
    result = schedule_circuit(routed, target, method=method)
    assert result.total_duration_s == pytest.approx(
        routed.estimate_duration(target, unit="s"), rel=1e-12
    )


def test_only_touched_qubits_get_timelines(routed, target):
    result = schedule_circuit(routed, target)
    assert 0 < len(result.timelines) <= 8  # 4 used + ancillas are untouched
    assert len(result.timelines) < target.num_qubits


def test_idle_windows_are_positive_and_inside_the_circuit(routed, target):
    result = schedule_circuit(routed, target)
    for _qubit, duration in result.idle_windows():
        assert duration > 0
        assert duration <= result.total_duration_s


def test_no_idle_is_charged_before_a_qubit_is_first_touched(routed, target):
    """A qubit sitting in |0> has nothing to decohere; leading gaps must not count."""
    result = schedule_circuit(routed, target)
    for timeline in result.timelines.values():
        first_busy_start = timeline.busy[0][0]
        for start, _stop in timeline.idle:
            assert start >= first_busy_start


def test_idle_windows_do_not_overlap_busy_windows(routed, target):
    result = schedule_circuit(routed, target)
    for timeline in result.timelines.values():
        for istart, istop in timeline.idle:
            for bstart, bstop in timeline.busy:
                assert istop <= bstart or istart >= bstop


def test_alap_never_charges_more_idle_than_asap(routed, target):
    """ALAP pushes gates late, moving idle into the (uncharged) prologue."""
    asap = schedule_circuit(routed, target, method="asap")
    alap = schedule_circuit(routed, target, method="alap")
    assert alap.total_idle_s() <= asap.total_idle_s() + 1e-15


def test_unknown_method_rejected(routed, target):
    with pytest.raises(TranspileAuditError):
        schedule_circuit(routed, target, method="sideways")


def test_serial_circuit_has_a_measurable_tail_idle(fake_backend, target):
    """The canonical case: qubit 0 idles while a CX cascade walks down the line,
    then idles again before measurement."""
    qc = QuantumCircuit(5)
    qc.h(0)
    for i in range(4):
        qc.cx(i, i + 1)
    qc.measure_all()
    pm = generate_preset_pass_manager(
        optimization_level=1, backend=fake_backend, seed_transpiler=11
    )
    result = schedule_circuit(pm.run(qc), target)
    assert result.total_idle_s() > 0
    assert max(d for _, d in result.idle_windows()) > 1e-7


def test_durations_come_from_the_target_not_a_hard_coded_list(routed, target):
    """rz and barrier must resolve to 0 via the target, not a hard-coded set."""
    result = schedule_circuit(routed, target)
    assert result.unknown_durations == ()


def test_untimed_instruction_raises_a_typed_error(target):
    """Qiskit's own message is opaque; we must translate, not leak it."""
    from qiskit.circuit import Gate

    qc = QuantumCircuit(target.num_qubits)
    qc.sx(0)
    qc.append(Gate("mystery_gate", 1, []), [0])
    qc.sx(0)
    with pytest.raises(TranspileAuditError) as exc:
        schedule_circuit(qc, target)
    assert "mystery_gate" in exc.value.message
    assert "ISA-valid" in (exc.value.hint or "")


def test_wrong_direction_two_qubit_gate_is_caught(snapshot, target):
    """A 2q gate on the uncalibrated orientation of a coupler must not pass."""
    edge = snapshot.edges[0]
    qc = QuantumCircuit(target.num_qubits)
    # Deliberately reversed relative to the calibrated direction.
    qc.ecr(edge.target, edge.control)
    with pytest.raises(TranspileAuditError):
        schedule_circuit(qc, target)
