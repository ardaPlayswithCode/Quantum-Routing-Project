"""Pydantic data models.

Deliberately Qiskit-free: nothing here holds a ``Target``, ``Backend`` or any
other Qiskit object.  Qiskit objects are not stable across versions and are not
JSON-serialisable, so caching them would be a correctness and portability trap.
A ``CalibrationSnapshot`` is a plain, round-trippable record of numbers.
"""

from __future__ import annotations

import datetime as _dt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .physics import clamp_t2

CalibrationSource = Literal["fake_backend", "cache", "live", "synthetic"]


class QubitCal(BaseModel):
    """Per-qubit calibration. All times in seconds, all errors dimensionless."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    t1: float | None = None
    t2: float | None = None
    readout_error: float | None = None
    sx_error: float | None = None
    frequency: float | None = None

    @field_validator("t1", "t2")
    @classmethod
    def _non_negative_time(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("coherence times must be non-negative")
        return v

    @property
    def static_error(self) -> float:
        """1q + readout error, the decoherence-free part of a qubit's badness."""
        return (self.sx_error or 0.0) + (self.readout_error or 0.0)

    @property
    def is_complete(self) -> bool:
        return None not in (self.t1, self.t2, self.readout_error, self.sx_error)


class EdgeCal(BaseModel):
    """Per-coupler calibration for one direction of a two-qubit gate."""

    model_config = ConfigDict(frozen=True)

    control: int = Field(ge=0)
    target: int = Field(ge=0)
    gate: str
    error: float | None = None
    duration: float | None = None

    @property
    def key(self) -> tuple[int, int]:
        return (self.control, self.target)


class CalibrationSnapshot(BaseModel):
    """A frozen, serialisable view of one backend's calibration at one instant."""

    model_config = ConfigDict(frozen=True)

    backend_name: str
    source: CalibrationSource
    captured_at: _dt.datetime
    num_qubits: int
    dt: float | None = None
    basis_gates: list[str] = Field(default_factory=list)
    qubits: list[QubitCal]
    edges: list[EdgeCal]
    # Undirected coupler list, useful for topology rendering.
    coupling: list[tuple[int, int]] = Field(default_factory=list)

    # Qiskit's MetaPass hashes every __init__ argument of a transpiler pass
    # (frozenset(arguments)).  Pydantic's default frozen-model hash walks
    # __dict__, which here contains lists and blows up.  Hash on stable
    # scalars instead so a snapshot can be handed to a pass constructor.
    def __hash__(self) -> int:  # type: ignore[override]
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

    # ---------------- lookups ----------------

    def qubit(self, index: int) -> QubitCal:
        return self._qubit_index()[index]

    def _qubit_index(self) -> dict[int, QubitCal]:
        cached = self.__dict__.get("_qmap")
        if cached is None:
            cached = {q.index: q for q in self.qubits}
            object.__setattr__(self, "_qmap", cached)
        return cached

    def _edge_index(self) -> dict[tuple[int, int], EdgeCal]:
        cached = self.__dict__.get("_emap")
        if cached is None:
            cached = {e.key: e for e in self.edges}
            object.__setattr__(self, "_emap", cached)
        return cached

    def edge(self, control: int, target: int) -> EdgeCal | None:
        """Directed lookup with an undirected fallback.

        Calibration usually reports one direction per coupler; a routed circuit
        may use either, and the physical error is symmetric to within
        calibration noise.
        """
        emap = self._edge_index()
        hit = emap.get((control, target))
        if hit is not None:
            return hit
        return emap.get((target, control))

    def t1(self, index: int) -> float | None:
        return self.qubit(index).t1

    def t2(self, index: int) -> float | None:
        return self.qubit(index).t2

    # ---------------- aggregate stats ----------------

    def median_two_qubit_duration(self) -> float:
        durations = [e.duration for e in self.edges if e.duration]
        if not durations:
            return 0.0
        durations.sort()
        n = len(durations)
        return durations[n // 2] if n % 2 else 0.5 * (durations[n // 2 - 1] + durations[n // 2])

    def age(self, now: _dt.datetime | None = None) -> _dt.timedelta:
        now = now or _dt.datetime.now(_dt.timezone.utc)
        captured = self.captured_at
        if captured.tzinfo is None:
            captured = captured.replace(tzinfo=_dt.timezone.utc)
        return now - captured

    def age_hours(self, now: _dt.datetime | None = None) -> float:
        return self.age(now).total_seconds() / 3600.0

    def missing_fields(self) -> list[str]:
        """Human-readable list of calibration holes, for the report header."""
        problems: list[str] = []
        for q in self.qubits:
            for field in ("t1", "t2", "readout_error", "sx_error"):
                if getattr(q, field) is None:
                    problems.append(f"qubit[{q.index}].{field}")
        for e in self.edges:
            if e.error is None:
                problems.append(f"edge[{e.control}->{e.target}].error")
            if e.duration is None:
                problems.append(f"edge[{e.control}->{e.target}].duration")
        return problems


class MetricSet(BaseModel):
    """Everything measured about one transpiled circuit."""

    model_config = ConfigDict(frozen=True)

    label: str
    seed: int
    layout: list[int]
    depth: int
    size: int
    two_qubit_gates: int
    swap_gates: int
    duration_s: float
    esp: float
    esp_gate_only: float
    esp_idle_only: float
    total_idle_s: float
    worst_t2_on_layout_s: float | None
    worst_t1_on_layout_s: float | None
    mean_t2_on_layout_s: float | None
    worst_edge_error: float | None
    readout_error_sum: float
    # populated only when the caller ran the Aer harness
    hellinger_fidelity: float | None = None


class AuditReport(BaseModel):
    """Top-level JSON payload emitted by ``q-audit run --json``."""

    schema_version: str
    tool_version: str
    generated_at: _dt.datetime
    circuit: dict
    backend: dict
    calibration: dict
    settings: dict
    baseline: MetricSet
    baseline_default_seed: MetricSet
    audited: MetricSet
    comparison: dict
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def clamp_snapshot_t2(qubits: list[QubitCal]) -> list[QubitCal]:
    """Apply the T2 <= 2*T1 bound to a freshly-ingested qubit list."""
    out: list[QubitCal] = []
    for q in qubits:
        t2 = clamp_t2(q.t1, q.t2)
        out.append(q if t2 == q.t2 else q.model_copy(update={"t2": t2}))
    return out
