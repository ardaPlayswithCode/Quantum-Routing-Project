"""Calibration ingestion, normalisation and on-disk caching.

Resolution order (first hit wins):
    1. ``--backend fake_<name>``  -> offline reference snapshot from
       ``qiskit_ibm_runtime.fake_provider``.  Fully deterministic, no network.
    2. on-disk JSON cache under ``platformdirs.user_cache_dir`` (1 hour TTL)
    3. live ``QiskitRuntimeService``

We never cache Qiskit objects.  The cache holds ``CalibrationSnapshot`` JSON --
plain numbers that survive a Qiskit upgrade.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir

from .errors import CalibrationError, StaleCalibrationError
from .models import CalibrationSnapshot, EdgeCal, QubitCal, clamp_snapshot_t2

CACHE_TTL_SECONDS = 3600  # 1 hour
STALE_LIMIT_HOURS = 24.0
_ONE_QUBIT_ERROR_PREFERENCE = ("sx", "x", "rx", "sxdg", "u", "u3")


def cache_dir() -> Path:
    override = os.environ.get("Q_AUDIT_CACHE_DIR")
    path = Path(override) if override else Path(user_cache_dir("q-audit", "q-audit"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_path(backend_name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in backend_name)
    return cache_dir() / f"calibration-{safe}.json"


# --------------------------------------------------------------------------
# Target -> snapshot
# --------------------------------------------------------------------------


def _two_qubit_op_names(target: Any) -> list[str]:
    names: list[str] = []
    for name in target.operation_names:
        try:
            props = target[name]
        except (KeyError, TypeError):
            continue
        if not props:
            continue
        for qargs in props:
            if qargs is not None and len(qargs) == 2:
                names.append(name)
            break
    return names


def _one_qubit_error(target: Any, qubit: int) -> float | None:
    for name in _ONE_QUBIT_ERROR_PREFERENCE:
        if name not in target.operation_names:
            continue
        try:
            props = target[name].get((qubit,))
        except (KeyError, TypeError):
            continue
        if props is not None and props.error is not None:
            return float(props.error)
    return None


def _readout_error(target: Any, qubit: int) -> float | None:
    if "measure" not in target.operation_names:
        return None
    try:
        props = target["measure"].get((qubit,))
    except (KeyError, TypeError):
        return None
    return None if props is None or props.error is None else float(props.error)


def snapshot_from_target(
    target: Any,
    *,
    backend_name: str,
    source: str,
    captured_at: _dt.datetime | None = None,
) -> CalibrationSnapshot:
    """Convert a Qiskit ``Target`` into a plain, serialisable snapshot."""
    captured_at = captured_at or _dt.datetime.now(_dt.timezone.utc)
    num_qubits = target.num_qubits
    qprops = target.qubit_properties or []

    qubits: list[QubitCal] = []
    for i in range(num_qubits):
        qp = qprops[i] if i < len(qprops) else None
        qubits.append(
            QubitCal(
                index=i,
                t1=None if qp is None else _as_float(qp.t1),
                t2=None if qp is None else _as_float(qp.t2),
                readout_error=_readout_error(target, i),
                sx_error=_one_qubit_error(target, i),
                frequency=None if qp is None else _as_float(getattr(qp, "frequency", None)),
            )
        )
    # Enforce T2 <= 2*T1 at the ingest boundary so nothing downstream has to.
    qubits = clamp_snapshot_t2(qubits)

    edges: list[EdgeCal] = []
    seen: set[tuple[int, int, str]] = set()
    for name in _two_qubit_op_names(target):
        for qargs, props in target[name].items():
            if qargs is None or len(qargs) != 2:
                continue
            key = (int(qargs[0]), int(qargs[1]), name)
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                EdgeCal(
                    control=int(qargs[0]),
                    target=int(qargs[1]),
                    gate=name,
                    error=None if props is None else _as_float(props.error),
                    duration=None if props is None else _as_float(props.duration),
                )
            )

    coupling: list[tuple[int, int]] = []
    cmap = target.build_coupling_map()
    if cmap is not None:
        undirected: set[tuple[int, int]] = set()
        for a, b in cmap.get_edges():
            undirected.add((min(a, b), max(a, b)))
        coupling = sorted(undirected)

    return CalibrationSnapshot(
        backend_name=backend_name,
        source=source,  # type: ignore[arg-type]
        captured_at=captured_at,
        num_qubits=num_qubits,
        dt=_as_float(getattr(target, "dt", None)),
        basis_gates=sorted(str(n) for n in target.operation_names),
        qubits=qubits,
        edges=edges,
        coupling=coupling,
    )


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # drop NaN


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def _fake_backend_class(name: str):
    """Map ``fake_sherbrooke`` -> ``FakeSherbrooke`` in the runtime fake provider."""
    from qiskit_ibm_runtime import fake_provider

    normalised = name.strip().lower().replace("-", "_")
    if not normalised.startswith("fake_"):
        normalised = "fake_" + normalised
    wanted = "fake" + normalised[len("fake_") :].replace("_", "")
    for attr in dir(fake_provider):
        if not attr.startswith("Fake"):
            continue
        if attr.lower() == wanted:
            return getattr(fake_provider, attr)
    available = sorted(
        "fake_" + a[4:].lower() for a in dir(fake_provider) if a.startswith("Fake")
    )
    raise CalibrationError(
        f"Unknown fake backend {name!r}.",
        hint="Try one of: " + ", ".join(available[:12]) + ", ...",
    )


def load_fake_backend(name: str) -> tuple[CalibrationSnapshot, Any]:
    """Load an offline reference snapshot and the backend object it came from.

    The backend object is returned because the transpiler needs a real
    ``Target``; the snapshot is what every *numeric* decision is made from.
    """
    cls = _fake_backend_class(name)
    backend = cls()
    captured = None
    try:
        props = backend.properties()
        raw = getattr(props, "last_update_date", None)
        if raw is not None:
            captured = raw if isinstance(raw, _dt.datetime) else None
    except Exception:  # noqa: BLE001 - properties() is best-effort metadata
        captured = None
    if captured is not None and captured.tzinfo is None:
        captured = captured.replace(tzinfo=_dt.timezone.utc)
    snap = snapshot_from_target(
        backend.target,
        backend_name=backend.name,
        source="fake_backend",
        captured_at=captured or _dt.datetime.now(_dt.timezone.utc),
    )
    return snap, backend


def load_from_cache(backend_name: str, *, ttl_seconds: int = CACHE_TTL_SECONDS) -> CalibrationSnapshot | None:
    path = cache_path(backend_name)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    try:
        snap = CalibrationSnapshot.model_validate(raw)
    except Exception:  # noqa: BLE001 - a stale schema is a cache miss, not a crash
        return None
    if snap.age().total_seconds() > ttl_seconds:
        return None
    return snap.model_copy(update={"source": "cache"})


def save_to_cache(snapshot: CalibrationSnapshot) -> Path:
    path = cache_path(snapshot.backend_name)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(snapshot.model_dump_json(indent=2))
    tmp.replace(path)
    return path


def load_live(backend_name: str) -> tuple[CalibrationSnapshot, Any]:
    """Fetch calibration from IBM Quantum. Requires saved credentials."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise CalibrationError("qiskit-ibm-runtime is not installed.") from exc
    try:
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
    except Exception as exc:  # noqa: BLE001 - network/auth failures are all the same to us
        raise CalibrationError(
            f"Could not reach IBM Quantum for backend {backend_name!r}: {exc}",
            hint="Run with --backend fake_sherbrooke to work offline, or save "
            "credentials with QiskitRuntimeService.save_account(...).",
        ) from exc
    snap = snapshot_from_target(backend.target, backend_name=backend_name, source="live")
    return snap, backend


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


def resolve_calibration(
    backend_name: str,
    *,
    use_cache: bool = True,
    refresh: bool = False,
) -> tuple[CalibrationSnapshot, Any]:
    """Resolve a calibration snapshot plus a Qiskit backend to transpile against.

    Returns ``(snapshot, backend)``.  ``backend`` may be ``None`` only if a
    cached snapshot were usable without one, which never happens today: the
    transpiler always needs a ``Target``, so a cache hit still instantiates the
    backend.  We keep the snapshot as the source of truth for every number and
    use the backend purely for its coupling graph and gate set.
    """
    if backend_name.lower().startswith("fake"):
        return load_fake_backend(backend_name)

    if use_cache and not refresh:
        cached = load_from_cache(backend_name)
        if cached is not None:
            _, backend = load_live(backend_name)
            return cached, backend

    snap, backend = load_live(backend_name)
    try:
        save_to_cache(snap)
    except OSError:
        pass
    return snap, backend


def check_staleness(
    snapshot: CalibrationSnapshot,
    *,
    allow_stale: bool,
    limit_hours: float = STALE_LIMIT_HOURS,
) -> list[str]:
    """Enforce the freshness policy. Returns warnings; raises when it must.

    Live and cached snapshots older than ``limit_hours`` are refused unless
    ``--allow-stale`` is passed -- recommending a layout from day-old T1/T2 is
    worse than recommending nothing.

    Offline *reference* snapshots (fake backends) are never refused: being
    frozen is the point of them.  They always carry a loud warning instead.
    """
    age_h = snapshot.age_hours()
    warnings: list[str] = []
    if snapshot.source == "fake_backend":
        warnings.append(
            f"Offline reference snapshot: {snapshot.backend_name} calibration is dated "
            f"{snapshot.captured_at.date()} ({age_h / 24.0:.0f} days old). Results "
            f"describe that frozen calibration, not today's hardware."
        )
        return warnings
    if age_h > limit_hours:
        if not allow_stale:
            raise StaleCalibrationError(
                f"Calibration for {snapshot.backend_name} is {age_h:.1f}h old "
                f"(limit {limit_hours:.0f}h); refusing to make a recommendation.",
                hint="Re-run with --allow-stale to proceed anyway, or --refresh to refetch.",
            )
        warnings.append(
            f"STALE CALIBRATION ACCEPTED: {age_h:.1f}h old (limit {limit_hours:.0f}h). "
            "Layout recommendations may be wrong."
        )
    return warnings
