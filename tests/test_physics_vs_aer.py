"""check our idle formula against aer's own thermal relaxation channel.

if these ever disagree then either i broke the formula or aer changed their
model, and i want to know about it either way
"""

import pytest

from q_audit.physics import idle_infidelity

pytestmark = pytest.mark.aer

CHANNELS = [
    # (T1, T2, idle). short and long idles, T2<T1, T2>T1, and one horrible
    # low T2 qubit like sherbrooke's q84
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
    """our clamp should do the same thing aer does internally"""
    t1, t = 100e-6, 4e-6
    ours = idle_infidelity(t, t1, 10 * t1)
    theirs = aer_infidelity(t1, 2 * t1, t)
    assert ours == pytest.approx(theirs, abs=1e-12)
