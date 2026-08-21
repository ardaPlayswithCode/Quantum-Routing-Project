"""Validate the analytic idle-infidelity formula against Aer's own channel.

If these ever disagree, either our formula drifted or Aer changed its
thermal-relaxation model -- both are things we want to hear about loudly.
"""

from __future__ import annotations

import pytest

from q_audit.physics import idle_infidelity

pytestmark = pytest.mark.aer

CHANNELS = [
    # (T1, T2, idle time) -- spans short/long idles, T2<T1, T2>T1, and a
    # pathological low-T2 qubit like fake_sherbrooke's q84.
    (100e-6, 80e-6, 1e-6),
    (100e-6, 80e-6, 10e-6),
    (100e-6, 150e-6, 5e-6),
    (50e-6, 50e-6, 50e-6),
    (300e-6, 2.6e-6, 2e-6),
    (200e-6, 400e-6, 1e-6),
    (100e-6, 80e-6, 1e-3),
    (514e-6, 489e-6, 3.3e-7),
]


@pytest.fixture(scope="module")
def aer_infidelity():
    aer_noise = pytest.importorskip("qiskit_aer.noise")
    from qiskit.quantum_info import SuperOp, average_gate_fidelity

    def _compute(t1: float, t2: float, t: float) -> float:
        error = aer_noise.thermal_relaxation_error(
            t1, t2, t, excited_state_population=0.0
        )
        return 1.0 - average_gate_fidelity(SuperOp(error.to_quantumchannel()))

    return _compute


@pytest.mark.parametrize(("t1", "t2", "t"), CHANNELS)
def test_matches_aer_thermal_relaxation(aer_infidelity, t1, t2, t):
    assert idle_infidelity(t, t1, t2) == pytest.approx(
        aer_infidelity(t1, t2, t), abs=1e-12
    )


def test_matches_aer_after_t2_clamping(aer_infidelity):
    """Our clamp must reproduce Aer's own T2 <= 2*T1 behaviour."""
    t1, t = 100e-6, 4e-6
    ours = idle_infidelity(t, t1, 10 * t1)
    theirs = aer_infidelity(t1, 2 * t1, t)
    assert ours == pytest.approx(theirs, abs=1e-12)
