"""regression gate against benchmarks/golden.json.

golden.json is what the audit produced on a run i know was good. this redoes a
small fast subset of it and complains if anything drops more than 2%. skips
rather than fails if the file isn't there, so a fresh clone doesn't have to run
the benchmark before it can run the tests.

regenerate with: python benchmarks/bench.py --seeds 20 --shots 16384 --write-golden
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

GOLDEN = Path(__file__).resolve().parents[1] / "benchmarks" / "golden.json"
REGRESSION_TOLERANCE = 0.02  # 2%, anything worse than that is a real regression

# only two circuits, this is a gate and not the benchmark itself
CHECKED = ["ghz7", "qft_mirror7"]


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip(f"{GOLDEN.name} not present; run benchmarks/bench.py --write-golden")
    return json.loads(GOLDEN.read_text())


def _reproduce(name: str, seeds: int, snapshot, backend):
    import sys

    sys.path.insert(0, str(GOLDEN.parents[1]))
    from benchmarks.circuits import SUITE

    from q_audit.feature_extract import extract_features
    from q_audit.transpile_audit import AuditSettings, run_audit

    circuit = SUITE[name].build()
    bound, unrolled, features = extract_features(circuit, snapshot)
    return run_audit(
        bound, unrolled, features, snapshot, backend, AuditSettings(seeds=seeds)
    )


@pytest.mark.parametrize("name", CHECKED)
def test_no_esp_regression_against_golden(name, golden, snapshot, fake_backend):
    key = f"{name}|alpha=1.0|normalized"
    if key not in golden["cases"]:
        pytest.skip(f"{key} not in golden file")
    expected = golden["cases"][key]
    result = _reproduce(name, golden["seeds"], snapshot, fake_backend)

    got = result.audited.best.esp
    want = expected["esp_audited"]
    assert got >= want * (1 - REGRESSION_TOLERANCE), (
        f"{name}: audited ESP regressed {100 * (got / want - 1):.2f}% "
        f"({want:.5f} -> {got:.5f})"
    )
    # the baseline is qiskit's own number, not ours. if that moved then the
    # environment changed and golden.json needs regenerating
    assert result.baseline.best.esp == pytest.approx(
        expected["esp_baseline"], rel=1e-6
    ), "baseline ESP moved: Qiskit or the calibration snapshot changed"


@pytest.mark.parametrize("name", CHECKED)
def test_audited_still_beats_baseline(name, golden, snapshot, fake_backend):
    key = f"{name}|alpha=1.0|normalized"
    if key not in golden["cases"]:
        pytest.skip(f"{key} not in golden file")
    expected = golden["cases"][key]
    if expected["esp_gain_pct"] is None or expected["esp_gain_pct"] <= 0:
        pytest.skip(f"{name} was not a win in the golden run either")
    result = _reproduce(name, golden["seeds"], snapshot, fake_backend)
    assert result.audited.best.esp > result.baseline.best.esp, (
        f"{name} used to win by {expected['esp_gain_pct']:.2f}% and now does not"
    )


@pytest.mark.parametrize("name", CHECKED)
def test_layout_strategy_is_stable(name, golden, snapshot, fake_backend):
    key = f"{name}|alpha=1.0|normalized"
    if key not in golden["cases"]:
        pytest.skip(f"{key} not in golden file")
    result = _reproduce(name, golden["seeds"], snapshot, fake_backend)
    assert result.audited_strategy == golden["cases"][key]["layout_source"]
