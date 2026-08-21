"""Calibration ingestion, the T2 clamp, the cache and the staleness policy."""

from __future__ import annotations

import datetime as _dt

import pytest

from q_audit.calibration import (
    CACHE_TTL_SECONDS,
    check_staleness,
    load_fake_backend,
    load_from_cache,
    save_to_cache,
    snapshot_from_target,
)
from q_audit.errors import CalibrationError, StaleCalibrationError
from q_audit.models import CalibrationSnapshot, EdgeCal, QubitCal, clamp_snapshot_t2

SHERBROOKE_QUBITS = 127
SHERBROOKE_COUPLERS = 144


def test_fake_sherbrooke_shape(snapshot):
    assert snapshot.num_qubits == SHERBROOKE_QUBITS
    assert len(snapshot.qubits) == SHERBROOKE_QUBITS
    assert len(snapshot.edges) == SHERBROOKE_COUPLERS
    assert len(snapshot.coupling) == SHERBROOKE_COUPLERS


def test_fake_sherbrooke_has_no_missing_values(snapshot):
    assert snapshot.missing_fields() == []
    assert all(q.is_complete for q in snapshot.qubits)


def test_all_values_are_physical(snapshot):
    for q in snapshot.qubits:
        assert q.t1 > 0
        assert q.t2 > 0
        assert q.t2 <= 2 * q.t1 + 1e-18, f"T2 clamp violated on qubit {q.index}"
        assert 0.0 <= q.readout_error <= 1.0
        assert 0.0 <= q.sx_error <= 1.0
    for e in snapshot.edges:
        assert 0.0 <= e.error <= 1.0
        assert e.duration > 0


def test_edge_lookup_is_direction_tolerant(snapshot):
    edge = snapshot.edges[0]
    assert snapshot.edge(edge.control, edge.target) is not None
    assert snapshot.edge(edge.target, edge.control) is not None
    assert snapshot.edge(999, 998) is None


def test_median_two_qubit_duration_is_sane(snapshot):
    median = snapshot.median_two_qubit_duration()
    assert 1e-7 < median < 1e-5  # hundreds of nanoseconds


def test_t2_clamp_applied_on_ingest():
    raw = [QubitCal(index=0, t1=100e-6, t2=900e-6, readout_error=0.01, sx_error=1e-4)]
    clamped = clamp_snapshot_t2(raw)
    assert clamped[0].t2 == 200e-6
    assert clamped[0].t1 == 100e-6


def test_snapshot_json_round_trip(snapshot):
    restored = CalibrationSnapshot.model_validate_json(snapshot.model_dump_json())
    assert restored.qubits == snapshot.qubits
    assert restored.edges == snapshot.edges
    assert restored.num_qubits == snapshot.num_qubits


def test_snapshot_is_hashable(snapshot):
    """Qiskit's MetaPass hashes every pass __init__ argument."""
    assert isinstance(hash(snapshot), int)


def test_fake_backend_reports_its_true_capture_date():
    snap, backend = load_fake_backend("fake_sherbrooke")
    assert snap.source == "fake_backend"
    assert backend.num_qubits == SHERBROOKE_QUBITS
    # The shipped calibration is a frozen file, not "now".
    assert snap.captured_at.year <= _dt.datetime.now(_dt.timezone.utc).year
    assert snap.age_hours() > 0


def test_fake_backend_name_normalisation():
    for name in ("fake_sherbrooke", "FAKE_SHERBROOKE", "sherbrooke", "fake-sherbrooke"):
        snap, _ = load_fake_backend(name)
        assert snap.backend_name == "fake_sherbrooke"


def test_unknown_fake_backend_raises_with_suggestions():
    with pytest.raises(CalibrationError) as exc:
        load_fake_backend("fake_not_a_real_machine")
    assert "Unknown fake backend" in exc.value.message
    assert exc.value.hint


def test_cache_round_trip_and_ttl(snapshot, tmp_path, monkeypatch):
    monkeypatch.setenv("Q_AUDIT_CACHE_DIR", str(tmp_path))
    fresh = snapshot.model_copy(
        update={
            "backend_name": "cache_probe",
            "source": "live",
            "captured_at": _dt.datetime.now(_dt.timezone.utc),
        }
    )
    save_to_cache(fresh)
    hit = load_from_cache("cache_probe")
    assert hit is not None
    assert hit.source == "cache"
    assert len(hit.qubits) == SHERBROOKE_QUBITS

    stale = fresh.model_copy(
        update={
            "captured_at": _dt.datetime.now(_dt.timezone.utc)
            - _dt.timedelta(seconds=CACHE_TTL_SECONDS + 60)
        }
    )
    save_to_cache(stale)
    assert load_from_cache("cache_probe") is None, "TTL not enforced"


def test_cache_miss_on_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.setenv("Q_AUDIT_CACHE_DIR", str(tmp_path))
    (tmp_path / "calibration-broken.json").write_text("{not json")
    assert load_from_cache("broken") is None


def test_staleness_refuses_old_live_data(snapshot):
    old = snapshot.model_copy(
        update={
            "source": "live",
            "captured_at": _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=30),
        }
    )
    with pytest.raises(StaleCalibrationError) as exc:
        check_staleness(old, allow_stale=False)
    assert exc.value.exit_code == 3
    warnings = check_staleness(old, allow_stale=True)
    assert any("STALE" in w for w in warnings)


def test_staleness_accepts_fresh_live_data(snapshot):
    fresh = snapshot.model_copy(
        update={"source": "live", "captured_at": _dt.datetime.now(_dt.timezone.utc)}
    )
    assert check_staleness(fresh, allow_stale=False) == []


def test_staleness_warns_but_never_refuses_a_reference_snapshot(snapshot):
    """A fake backend is frozen by design; refusing it would break offline use."""
    warnings = check_staleness(snapshot, allow_stale=False)
    assert len(warnings) == 1
    assert "Offline reference snapshot" in warnings[0]


def test_snapshot_from_target_handles_missing_qubit_properties(target):
    snap = snapshot_from_target(target, backend_name="probe", source="live")
    assert snap.backend_name == "probe"
    assert snap.source == "live"
    assert snap.dt and snap.dt > 0
    assert "ecr" in snap.basis_gates
