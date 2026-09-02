"""calibration data: the models, where we get them from, and the disk cache.

we look for calibration in this order: fake backend (offline, no network),
then the json cache on disk, then live ibm.

the models used to live in their own models.py but they're 90% calibration
numbers so i moved them here.

one rule: never cache qiskit objects. a Target isn't json and doesn't survive
a qiskit upgrade. the snapshot is just floats, so it does.
"""

import datetime as _dt
import json
import os
from pathlib import Path

from platformdirs import user_cache_dir
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .errors import CalibrationError, StaleCalibrationError
from .physics import clamp_t2

CACHE_TTL_SECONDS = 3600  # 1 hour
STALE_LIMIT_HOURS = 24.0
# which 1q gate to believe for the per qubit error, first one that exists wins
_ONE_QUBIT_ERROR_PREFERENCE = ("sx", "x", "rx", "sxdg", "u", "u3")


# ---------------- the models ----------------


class QubitCal(BaseModel):
    """one qubit's numbers. times in seconds, errors are just fractions"""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    t1: float | None = None
    t2: float | None = None
    readout_error: float | None = None
    sx_error: float | None = None
    frequency: float | None = None

    @field_validator("t1", "t2")
    @classmethod
    def _non_negative_time(cls, v):
        if v is not None and v < 0:
            raise ValueError("coherence times must be non-negative")
        return v

    @property
    def static_error(self):
        """the part of a qubit's badness that has nothing to do with idling"""
        return (self.sx_error or 0.0) + (self.readout_error or 0.0)

    @property
    def is_complete(self):
        return None not in (self.t1, self.t2, self.readout_error, self.sx_error)


class EdgeCal(BaseModel):
    """one direction of one coupler"""

    model_config = ConfigDict(frozen=True)

    control: int = Field(ge=0)
    target: int = Field(ge=0)
    gate: str
    error: float | None = None
    duration: float | None = None

    @property
    def key(self):
        return (self.control, self.target)


class CalibrationSnapshot(BaseModel):
    """one backend's calibration, frozen at one moment, all plain numbers"""

    model_config = ConfigDict(frozen=True)

    backend_name: str
    source: str
    captured_at: _dt.datetime
    num_qubits: int
    dt: float | None = None
    basis_gates: list[str] = Field(default_factory=list)
    qubits: list[QubitCal]
    edges: list[EdgeCal]
    coupling: list[tuple[int, int]] = Field(default_factory=list)

    # took me ages to work out why passing a snapshot to a pass blew up:
    # qiskit's MetaPass hashes every __init__ argument, and pydantic's frozen
    # hash walks __dict__ which has lists in it. so hash the scalars instead.
    def __hash__(self):
        return hash(
            (
                self.backend_name,
                self.source,
                self.captured_at,
                self.num_qubits,
                len(self.qubits),
                len(self.edges),
            )
        )

    def qubit(self, index):
        return self._qubit_index()[index]

    def _qubit_index(self):
        # lazy dict so we're not doing a linear scan per lookup. frozen model
        # so it has to go in through object.__setattr__
        cached = self.__dict__.get("_qmap")
        if cached is None:
            cached = {q.index: q for q in self.qubits}
            object.__setattr__(self, "_qmap", cached)
        return cached

    def _edge_index(self):
        cached = self.__dict__.get("_emap")
        if cached is None:
            cached = {e.key: e for e in self.edges}
            object.__setattr__(self, "_emap", cached)
        return cached

    def edge(self, control, target):
        # calibration usually only gives one direction per coupler but a routed
        # circuit can use either, and physically it's the same coupler
        emap = self._edge_index()
        hit = emap.get((control, target))
        if hit is not None:
            return hit
        return emap.get((target, control))

    def t1(self, index):
        return self.qubit(index).t1

    def t2(self, index):
        return self.qubit(index).t2

    def median_two_qubit_duration(self):
        durations = [e.duration for e in self.edges if e.duration]
        if not durations:
            return 0.0
        durations.sort()
        n = len(durations)
        if n % 2:
            return durations[n // 2]
        return 0.5 * (durations[n // 2 - 1] + durations[n // 2])

    def age(self, now=None):
        now = now or _dt.datetime.now(_dt.timezone.utc)
        captured = self.captured_at
        if captured.tzinfo is None:
            captured = captured.replace(tzinfo=_dt.timezone.utc)
        return now - captured

    def age_hours(self, now=None):
        return self.age(now).total_seconds() / 3600.0

    def missing_fields(self):
        """list of holes in the calibration, for the report header"""
        problems = []
        for q in self.qubits:
            for name in ("t1", "t2", "readout_error", "sx_error"):
                if getattr(q, name) is None:
                    problems.append(f"qubit[{q.index}].{name}")
        for e in self.edges:
            if e.error is None:
                problems.append(f"edge[{e.control}->{e.target}].error")
            if e.duration is None:
                problems.append(f"edge[{e.control}->{e.target}].duration")
        return problems


def clamp_snapshot_t2(qubits):
    """apply T2 <= 2*T1 to a freshly loaded qubit list"""
    out = []
    for q in qubits:
        t2 = clamp_t2(q.t1, q.t2)
        out.append(q if t2 == q.t2 else q.model_copy(update={"t2": t2}))
    return out


# ---------------- cache paths ----------------


def cache_dir():
    override = os.environ.get("Q_AUDIT_CACHE_DIR")
    path = Path(override) if override else Path(user_cache_dir("q-audit", "q-audit"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_path(backend_name):
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in backend_name)
    return cache_dir() / f"calibration-{safe}.json"


# ---------------- Target -> snapshot ----------------


def _two_qubit_op_names(target):
    names = []
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


def _one_qubit_error(target, qubit):
    # T1 and T2 come from qubit_properties but the gate errors are stored
    # seperately per operation, so this has to go looking
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


def _readout_error(target, qubit):
    if "measure" not in target.operation_names:
        return None
    try:
        props = target["measure"].get((qubit,))
    except (KeyError, TypeError):
        return None
    if props is None or props.error is None:
        return None
    return float(props.error)


def snapshot_from_target(target, *, backend_name, source, captured_at=None):
    """pull a qiskit Target apart into a snapshot of plain numbers"""
    captured_at = captured_at or _dt.datetime.now(_dt.timezone.utc)
    num_qubits = target.num_qubits
    qprops = target.qubit_properties or []

    qubits = []
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
    # clamp here so nothing downstream has to think about it
    qubits = clamp_snapshot_t2(qubits)

    edges = []
    seen = set()
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

    coupling = []
    cmap = target.build_coupling_map()
    if cmap is not None:
        undirected = set()
        for a, b in cmap.get_edges():
            undirected.add((min(a, b), max(a, b)))
        coupling = sorted(undirected)

    return CalibrationSnapshot(
        backend_name=backend_name,
        source=source,
        captured_at=captured_at,
        num_qubits=num_qubits,
        dt=_as_float(getattr(target, "dt", None)),
        basis_gates=sorted(str(n) for n in target.operation_names),
        qubits=qubits,
        edges=edges,
        coupling=coupling,
    )


def _as_float(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # f != f catches NaN


# ---------------- where snapshots come from ----------------


def _fake_backend_class(name):
    """fake_sherbrooke -> FakeSherbrooke"""
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
        f"couldn't find a fake backend called {name!r}",
        hint="try one of: " + ", ".join(available[:12]) + ", ...",
    )


def load_fake_backend(name):
    """offline snapshot plus the backend object it came from.

    we hand back the backend too because the transpiler needs a real Target,
    but every number we actually use comes from the snapshot
    """
    cls = _fake_backend_class(name)
    backend = cls()
    captured = None
    try:
        props = backend.properties()
        raw = getattr(props, "last_update_date", None)
        if raw is not None:
            captured = raw if isinstance(raw, _dt.datetime) else None
    except Exception:  # properties() is just metadata, don't die over it
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


def load_from_cache(backend_name, *, ttl_seconds=CACHE_TTL_SECONDS):
    path = cache_path(backend_name)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    try:
        snap = CalibrationSnapshot.model_validate(raw)
    except Exception:  # old schema in the cache is a miss, not a crash
        return None
    if snap.age().total_seconds() > ttl_seconds:
        return None
    return snap.model_copy(update={"source": "cache"})


def save_to_cache(snapshot):
    path = cache_path(snapshot.backend_name)
    # write to a temp file first so a crash mid write can't leave junk
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(snapshot.model_dump_json(indent=2))
    tmp.replace(path)
    return path


def load_live(backend_name):
    """get calibration from ibm. needs saved credentials"""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise CalibrationError("qiskit-ibm-runtime isn't installed") from exc
    try:
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
    except Exception as exc:  # network, auth, whatever, same story
        raise CalibrationError(
            f"couldn't reach IBM Quantum for {backend_name!r}: {exc}",
            hint="use --backend fake_sherbrooke to work offline, or save "
            "credentials with QiskitRuntimeService.save_account(...)",
        ) from exc
    snap = snapshot_from_target(backend.target, backend_name=backend_name, source="live")
    return snap, backend


def resolve_calibration(backend_name, *, use_cache=True, refresh=False):
    """work out which snapshot to use and get a backend to transpile against.

    returns (snapshot, backend). even on a cache hit we still build the backend
    since the transpiler needs a Target, we just don't take its numbers
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


def check_staleness(snapshot, *, allow_stale, limit_hours=STALE_LIMIT_HOURS):
    """freshness policy. gives back warnings, raises if the data is too old.

    recommending a layout off day old T1/T2 is worse than recommending nothing.
    fake backends are exempt, being frozen is the whole point of them
    """
    age_h = snapshot.age_hours()
    warnings = []
    if snapshot.source == "fake_backend":
        warnings.append(
            f"offline reference snapshot: {snapshot.backend_name} calibration is from "
            f"{snapshot.captured_at.date()} ({age_h / 24.0:.0f} days ago). these numbers "
            f"describe that frozen calibration, not today's hardware."
        )
        return warnings
    if age_h > limit_hours:
        if not allow_stale:
            raise StaleCalibrationError(
                f"calibration for {snapshot.backend_name} is {age_h:.1f}h old "
                f"(limit is {limit_hours:.0f}h), not going to recommend a layout off that",
                hint="re-run with --allow-stale if you really want it, or --refresh to refetch",
            )
        warnings.append(
            f"STALE CALIBRATION ACCEPTED: {age_h:.1f}h old (limit {limit_hours:.0f}h). "
            "the layout recommendation might be wrong."
        )
    return warnings
