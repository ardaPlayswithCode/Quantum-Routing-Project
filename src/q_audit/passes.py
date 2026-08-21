"""The one place q-audit reaches into Qiskit's transpiler internals.

Why an ErrorMap and not a custom SABRE
--------------------------------------
SABRE's cost heuristic lives in Rust and takes no Python hook: there is no
supported way to make its swap selection noise-aware from Python.  VF2Layout
and VF2PostLayout, by contrast, read an optional ``vf2_avg_error_map`` out of
the property set and score candidate embeddings with it.  So the leverage point
is *layout*, not routing: we bias which physical qubits the circuit lands on,
then let stock SABRE route between them.

That hook is a private-ish contract.  ``tests/test_injection_contract.py`` is a
canary that fails loudly if a Qiskit upgrade changes it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.passes.layout.vf2_utils import ErrorMap

from .feature_extract import DagOpStats
from .models import CalibrationSnapshot
from .physics import idle_infidelity

VF2_ERROR_MAP_KEY = "vf2_avg_error_map"


# eq=False -> identity hash. Frozen dataclasses derive __hash__ from their
# fields, and these fields are dicts; a pass constructor argument must hash.
@dataclass(frozen=True, eq=False)
class ErrorMapBuild:
    """The injected map plus the raw numbers behind it, for reporting."""

    error_map: ErrorMap
    diagonal: dict[int, float]
    static: dict[int, float]
    idle_penalty: dict[int, float]
    t_est_s: float
    alpha: float
    scaling: str = "normalized"
    exponents: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "t_est_s": self.t_est_s,
            "alpha": self.alpha,
            "scaling": self.scaling,
            "exponents": self.exponents,
            "diagonal_median": (
                sorted(self.diagonal.values())[len(self.diagonal) // 2]
                if self.diagonal
                else None
            ),
            "worst_qubits": [
                {"qubit": q, "penalty": v} for q, v in self.worst_qubits(5)
            ],
            "best_qubits": [
                {"qubit": q, "penalty": v} for q, v in self.best_qubits(5)
            ],
        }

    def worst_qubits(self, n: int = 5) -> list[tuple[int, float]]:
        return sorted(self.diagonal.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def best_qubits(self, n: int = 5) -> list[tuple[int, float]]:
        return sorted(self.diagonal.items(), key=lambda kv: kv[1])[:n]


def build_decoherence_error_map(
    snapshot: CalibrationSnapshot,
    *,
    t_est_s: float,
    alpha: float = 1.0,
    op_stats: DagOpStats | None = None,
    scaling: str = "normalized",
) -> ErrorMapBuild:
    """Build a VF2 ``ErrorMap`` whose diagonal carries a T1/T2 idle penalty.

    Off-diagonal (per coupler) is the calibrated 2q gate error, passed through
    **unchanged**.  A reported 2q error already includes the decoherence that
    happens during the gate; adding a relaxation term on top would double count
    and would systematically over-punish the slow (but often high-fidelity)
    couplers.

    Diagonal, ``scaling="raw"`` -- the literal formula from the design doc::

        d[q] = sx_error[q] + readout_error[q] + alpha * eps_idle(t_est, T1, T2)

    Diagonal, ``scaling="normalized"`` (default) -- the same physics, corrected
    for how VF2 actually consumes the number.  VF2 scores a candidate embedding
    as ``prod_q (1 - d[q]) ** n1q(q)``, so ``d[q]`` is a *per-1q-operation*
    rate.  Readout happens once per qubit and the idle penalty is already a
    whole-circuit total, so both must be spread across the exponent::

        d[q] = 1 - exp( ( g_bar*ln(1-sx) + m_bar*ln(1-ro)
                          + ln(1 - alpha*eps_idle) ) / n_bar )

    with ``n_bar`` the mean 1q-op count per active qubit, ``g_bar`` the mean
    error-bearing 1q-gate count and ``m_bar`` the mean measurement count.  This
    makes the VF2 score a faithful proxy for ESP instead of one that scales the
    readout and idle terms by an arbitrary gate count.
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if scaling not in ("raw", "normalized"):
        raise ValueError(f"unknown scaling {scaling!r}; use 'raw' or 'normalized'")

    n_bar = op_stats.n_bar if op_stats else 0.0
    g_bar = op_stats.g_bar if op_stats else 0.0
    m_bar = op_stats.m_bar if op_stats else 0.0
    effective_scaling = scaling
    if scaling == "normalized" and n_bar <= 0.0:
        # Nothing to spread the one-shot terms over; the raw formula is the
        # only sensible thing left.
        effective_scaling = "raw"

    diagonal: dict[int, float] = {}
    static: dict[int, float] = {}
    idle_pen: dict[int, float] = {}

    for cal in snapshot.qubits:
        sx = min(0.999999, max(0.0, cal.sx_error or 0.0))
        ro = min(0.999999, max(0.0, cal.readout_error or 0.0))
        idle = idle_infidelity(t_est_s, cal.t1, cal.t2)
        static[cal.index] = sx + ro
        idle_pen[cal.index] = idle

        if effective_scaling == "raw":
            value = sx + ro + alpha * idle
        else:
            weighted_idle = min(0.999999, max(0.0, alpha * idle))
            log_fid = (
                g_bar * math.log1p(-sx)
                + m_bar * math.log1p(-ro)
                + math.log1p(-weighted_idle)
            )
            value = 1.0 - math.exp(log_fid / n_bar)
        # Clip into [0, 1]: VF2 scores in probability space, an error > 1 makes
        # the score meaningless rather than merely bad.
        diagonal[cal.index] = min(1.0, max(0.0, value))

    n_entries = len(diagonal) + 2 * len(snapshot.edges)
    error_map = ErrorMap(max(n_entries, 1))
    for q, err in diagonal.items():
        error_map.add_error((q, q), float(err))
    for edge in snapshot.edges:
        err = 1.0 if edge.error is None else min(1.0, max(0.0, float(edge.error)))
        # Both orientations: the interaction graph VF2 scores is undirected.
        error_map.add_error((edge.control, edge.target), err)
        error_map.add_error((edge.target, edge.control), err)

    return ErrorMapBuild(
        error_map=error_map,
        diagonal=diagonal,
        static=static,
        idle_penalty=idle_pen,
        t_est_s=t_est_s,
        alpha=alpha,
        scaling=effective_scaling,
        exponents={"n_bar": n_bar, "g_bar": g_bar, "m_bar": m_bar},
    )


class InjectDecoherenceErrorMap(AnalysisPass):
    """Put a T1/T2-aware ``ErrorMap`` where VF2Layout/VF2PostLayout will find it.

    Analysis-only: it never touches the DAG.  It must run *before* VF2Layout in
    the same pass manager (the property set is shared across stages of a
    ``StagedPassManager``, so one injection covers both VF2 passes).
    """

    def __init__(
        self,
        snapshot: CalibrationSnapshot,
        *,
        t_est_s: float,
        alpha: float = 1.0,
        op_stats: DagOpStats | None = None,
        scaling: str = "normalized",
        build: ErrorMapBuild | None = None,
    ) -> None:
        super().__init__()
        self.snapshot = snapshot
        self.t_est_s = t_est_s
        self.alpha = alpha
        self.build = build or build_decoherence_error_map(
            snapshot,
            t_est_s=t_est_s,
            alpha=alpha,
            op_stats=op_stats,
            scaling=scaling,
        )

    def run(self, dag):  # noqa: D102 - AnalysisPass contract
        self.property_set[VF2_ERROR_MAP_KEY] = self.build.error_map
        self.property_set["q_audit_error_map_build"] = self.build
        return dag


def error_map_is_supported() -> bool:
    """Cheap runtime check that the injection contract still holds.

    Verifies (a) ``ErrorMap`` still exposes ``add_error``/``get`` and (b) both
    VF2 passes still read the property-set key.  Used by the CLI to fail with a
    clear message instead of silently producing a no-op audit.
    """
    try:
        em = ErrorMap(1)
        em.add_error((0, 0), 0.1)
        if em.get((0, 0)) is None:
            return False
    except Exception:  # noqa: BLE001
        return False

    import inspect

    from qiskit.transpiler.passes.layout import vf2_layout, vf2_post_layout

    for module in (vf2_layout, vf2_post_layout):
        try:
            src = inspect.getsource(module)
        except OSError:
            return False
        if VF2_ERROR_MAP_KEY not in src:
            return False
    return True
