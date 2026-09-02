"""the one place we poke at qiskit's transpiler internals.

short version of why it works this way: sabre's swap heuristic is written in
rust and there's no python hook into it, so you can't make routing noise aware.
but VF2Layout and VF2PostLayout both look for an optional error map in the
property set under "vf2_avg_error_map" and score candidate layouts with it. so
we put our own map there. that means we pick better *qubits* and then let stock
sabre do the routing between them.

this key isn't really public api, which is why tests/test_injection_contract.py
exists. if a qiskit upgrade changes it the audit doesn't crash, it just quietly
does nothing, which is worse
"""

import math
from dataclasses import dataclass

from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.passes.layout.vf2_utils import ErrorMap

from .physics import idle_infidelity

VF2_ERROR_MAP_KEY = "vf2_avg_error_map"


# eq=False so it hashes by identity. frozen dataclasses build __hash__ out of
# their fields and these fields are dicts, and pass arguments have to hash
@dataclass(frozen=True, eq=False)
class ErrorMapBuild:
    """the map we inject plus the raw numbers behind it so we can report them"""

    error_map: ErrorMap
    diagonal: dict
    static: dict
    idle_penalty: dict
    t_est_s: float
    alpha: float
    scaling: str
    exponents: dict

    def to_dict(self):
        if self.diagonal:
            median = sorted(self.diagonal.values())[len(self.diagonal) // 2]
        else:
            median = None
        return {
            "t_est_s": self.t_est_s,
            "alpha": self.alpha,
            "scaling": self.scaling,
            "exponents": self.exponents,
            "diagonal_median": median,
            "worst_qubits": [{"qubit": q, "penalty": v} for q, v in self.worst_qubits(5)],
            "best_qubits": [{"qubit": q, "penalty": v} for q, v in self.best_qubits(5)],
        }

    def worst_qubits(self, n=5):
        return sorted(self.diagonal.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def best_qubits(self, n=5):
        return sorted(self.diagonal.items(), key=lambda kv: kv[1])[:n]


def build_decoherence_error_map(
    snapshot,
    *,
    t_est_s,
    alpha=1.0,
    op_stats=None,
    scaling="normalized",
):
    """build the error map. diagonal = per qubit, off diagonal = per coupler.

    the off diagonal is just the calibrated 2q error, untouched. a 2q error
    already has the decoherence during the gate baked in, adding relaxation on
    top would double count it and would punish the slow couplers, which are
    often the good ones.

    the diagonal is where the T1/T2 goes. with scaling="raw" it's literally

        d[q] = sx + readout + alpha * eps_idle(t_est, T1, T2)

    with scaling="normalized" (the default) it's the same physics but divided
    out by how vf2 actually uses the number. vf2 scores a layout as
    prod_q (1 - d[q]) ** n1q(q), so d[q] is a *per 1q op* rate, not a per qubit
    total. readout happens once and the idle penalty is already a whole circuit
    number, so both have to be spread over the exponent:

        d[q] = 1 - exp((g_bar*ln(1-sx) + m_bar*ln(1-ro)
                        + ln(1 - alpha*eps_idle)) / n_bar)

    took me a while to figure out that this is what was wrong with the raw
    version. normalized is worth about double in the benchmarks
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if scaling not in ("raw", "normalized"):
        raise ValueError(f"unknown scaling {scaling!r}, use 'raw' or 'normalized'")

    n_bar = op_stats.n_bar if op_stats else 0.0
    g_bar = op_stats.g_bar if op_stats else 0.0
    m_bar = op_stats.m_bar if op_stats else 0.0
    eff_scaling = scaling
    if scaling == "normalized" and n_bar <= 0.0:
        # nothing to divide by, fall back to raw
        eff_scaling = "raw"

    diagonal = {}
    static = {}
    idle_pen = {}

    for cal in snapshot.qubits:
        sx = min(0.999999, max(0.0, cal.sx_error or 0.0))
        ro = min(0.999999, max(0.0, cal.readout_error or 0.0))
        idle = idle_infidelity(t_est_s, cal.t1, cal.t2)
        static[cal.index] = sx + ro
        idle_pen[cal.index] = idle

        if eff_scaling == "raw":
            value = sx + ro + alpha * idle
        else:
            weighted_idle = min(0.999999, max(0.0, alpha * idle))
            log_fid = (
                g_bar * math.log1p(-sx)
                + m_bar * math.log1p(-ro)
                + math.log1p(-weighted_idle)
            )
            value = 1.0 - math.exp(log_fid / n_bar)
        # vf2 scores in probability space so an error above 1 makes the score
        # meaningless rather than just bad
        diagonal[cal.index] = min(1.0, max(0.0, value))

    n_entries = len(diagonal) + 2 * len(snapshot.edges)
    error_map = ErrorMap(max(n_entries, 1))
    for q, err in diagonal.items():
        error_map.add_error((q, q), float(err))
    for edge in snapshot.edges:
        if edge.error is None:
            err = 1.0
        else:
            err = min(1.0, max(0.0, float(edge.error)))
        # both orientations, the graph vf2 scores is undirected
        error_map.add_error((edge.control, edge.target), err)
        error_map.add_error((edge.target, edge.control), err)

    return ErrorMapBuild(
        error_map=error_map,
        diagonal=diagonal,
        static=static,
        idle_penalty=idle_pen,
        t_est_s=t_est_s,
        alpha=alpha,
        scaling=eff_scaling,
        exponents={"n_bar": n_bar, "g_bar": g_bar, "m_bar": m_bar},
    )


class InjectDecoherenceErrorMap(AnalysisPass):
    """drops our error map into the property set where vf2 will find it.

    analysis only, never touches the dag. has to run before VF2Layout in the
    same pass manager. the property set is shared across stages of a
    StagedPassManager so injecting once covers both vf2 passes
    """

    def __init__(self, snapshot, *, t_est_s, alpha=1.0, op_stats=None,
                 scaling="normalized", build=None):
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

    def run(self, dag):
        self.property_set[VF2_ERROR_MAP_KEY] = self.build.error_map
        self.property_set["q_audit_error_map_build"] = self.build
        return dag


def error_map_is_supported():
    """quick check that the hook is still there, so the cli can say so.

    checks ErrorMap still has add_error/get and that both vf2 passes still
    mention the property set key. grepping the source is gross but it's the
    only way to tell without actually running a transpile
    """
    try:
        em = ErrorMap(1)
        em.add_error((0, 0), 0.1)
        if em.get((0, 0)) is None:
            return False
    except Exception:
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
