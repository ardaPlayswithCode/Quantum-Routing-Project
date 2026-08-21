"""Pure-physics unit tests. No Qiskit, no backend, no network."""

from __future__ import annotations

import math

import pytest

from q_audit.physics import (
    clamp_t2,
    esp_from_terms,
    hellinger_fidelity,
    idle_infidelity,
    paired_mean_ci,
    percentile,
    survival_probability,
)

T1 = 100e-6
T2 = 80e-6


def test_idle_infidelity_is_zero_at_zero_time():
    assert idle_infidelity(0.0, T1, T2) == 0.0
    assert idle_infidelity(-1.0, T1, T2) == 0.0


def test_idle_infidelity_tends_to_zero_as_time_tends_to_zero():
    for t in (1e-12, 1e-11, 1e-10, 1e-9):
        assert idle_infidelity(t, T1, T2) < 1e-4
    assert idle_infidelity(1e-15, T1, T2) == pytest.approx(0.0, abs=1e-9)


def test_idle_infidelity_saturates_at_one_half():
    assert idle_infidelity(1.0, T1, T2) == pytest.approx(0.5, abs=1e-9)
    assert idle_infidelity(1e9, T1, T2) == 0.5
    # Never exceeds the maximally-mixed limit, whatever the inputs.
    for t in (1e-3, 1.0, 1e6):
        assert 0.0 <= idle_infidelity(t, T1, T2) <= 0.5


def test_idle_infidelity_is_monotone_in_time():
    previous = -1.0
    for t in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        current = idle_infidelity(t, T1, T2)
        assert current > previous
        previous = current


def test_idle_infidelity_is_monotone_in_coherence():
    """A qubit with better T2 must never be penalised more."""
    t = 5e-6
    worse = idle_infidelity(t, T1, 10e-6)
    better = idle_infidelity(t, T1, 200e-6)
    assert worse > better


def test_idle_infidelity_matches_closed_form():
    t = 3e-6
    expected = (3 - 2 * math.exp(-t / T2) - math.exp(-t / T1)) / 6
    assert idle_infidelity(t, T1, T2) == pytest.approx(expected, rel=1e-12)


def test_idle_infidelity_clamps_t2_internally():
    """T2 = 5*T1 must be treated as T2 = 2*T1, not taken at face value."""
    assert idle_infidelity(1e-6, T1, 5 * T1) == pytest.approx(
        idle_infidelity(1e-6, T1, 2 * T1), rel=1e-12
    )


def test_idle_infidelity_handles_missing_coherence_data():
    assert idle_infidelity(1e-6, None, None) == 0.5
    assert idle_infidelity(0.0, None, None) == 0.0
    assert idle_infidelity(1e-6, 0.0, 0.0) == 0.5
    # One good time is still information.
    assert 0.0 < idle_infidelity(1e-6, T1, None) < 0.5


def test_clamp_t2():
    assert clamp_t2(100e-6, 500e-6) == 200e-6
    assert clamp_t2(100e-6, 50e-6) == 50e-6
    assert clamp_t2(None, 50e-6) == 50e-6
    assert clamp_t2(100e-6, None) is None


def test_survival_probability():
    assert survival_probability(0.0, T1) == 1.0
    assert survival_probability(T1, T1) == pytest.approx(math.exp(-1))
    assert survival_probability(1e-6, None) == 0.0


def test_esp_multiplies_independent_terms():
    esp = esp_from_terms([0.01, 0.02], [])
    assert esp == pytest.approx(0.99 * 0.98, rel=1e-12)


def test_esp_includes_idle_terms():
    gate_only = esp_from_terms([0.01], [])
    with_idle = esp_from_terms([0.01], [(1e-5, T1, T2)])
    assert with_idle < gate_only


def test_esp_is_zero_for_a_dead_gate():
    assert esp_from_terms([1.0], []) == 0.0


def test_esp_does_not_underflow_to_nonsense():
    """Thousands of small terms must stay finite and in range."""
    esp = esp_from_terms([0.001] * 5000, [])
    assert 0.0 <= esp <= 1.0
    assert esp == pytest.approx(math.exp(5000 * math.log(0.999)), rel=1e-9)


def test_esp_of_nothing_is_one():
    assert esp_from_terms([], []) == 1.0


def test_hellinger_fidelity_identical_and_orthogonal():
    assert hellinger_fidelity({"00": 500, "11": 500}, {"00": 5, "11": 5}) == pytest.approx(1.0)
    assert hellinger_fidelity({"00": 1000}, {"11": 1000}) == pytest.approx(0.0)


def test_hellinger_fidelity_normalises_counts():
    a = {"0": 30, "1": 70}
    b = {"0": 3000, "1": 7000}
    assert hellinger_fidelity(a, b) == pytest.approx(1.0)


def test_percentile():
    values = [1, 2, 3, 4, 5]
    assert percentile(values, 0) == 1
    assert percentile(values, 50) == 3
    assert percentile(values, 100) == 5
    assert percentile(values, 25) == 2
    assert percentile([7], 42) == 7


def test_paired_mean_ci_brackets_the_mean():
    deltas = [0.01, 0.02, 0.015, 0.012, 0.018] * 4
    mean, lo, hi = paired_mean_ci(deltas)
    assert lo < mean < hi
    assert mean == pytest.approx(sum(deltas) / len(deltas))


def test_paired_mean_ci_degenerate_inputs():
    assert paired_mean_ci([]) == (0.0, 0.0, 0.0)
    assert paired_mean_ci([0.5]) == (0.5, 0.5, 0.5)
