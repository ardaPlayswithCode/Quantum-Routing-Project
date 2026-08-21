"""Pure-Python noise arithmetic. No Qiskit imports live here, by design.

Everything in this module is a plain function over floats so it can be unit
tested (and validated against Aer) without constructing a circuit.

Physical model
--------------
A qubit sitting idle for time ``t`` under amplitude damping ``T1`` and pure
dephasing folded into ``T2`` suffers an average gate infidelity of

    eps_idle(t) = (3 - 2*exp(-t/T2) - exp(-t/T1)) / 6

This is the standard average infidelity of a thermal-relaxation channel at zero
excited-state population.  Limits: ``eps(0) = 0`` and ``eps(inf) = 1/2``.

Decoherence is applied ONLY to idle windows.  Reported gate errors (from
backend calibration) already contain the decoherence accumulated *during* the
gate, so adding a relaxation term on top of a gate error would double count.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

# A T2 longer than 2*T1 is unphysical; hardware calibration occasionally reports
# it because T1 and T2 are measured in separate experiments.
T2_OVER_T1_LIMIT = 2.0

# Below this, treat a coherence time as "instantly decohered" rather than
# dividing by ~zero.
_MIN_COHERENCE_S = 1e-12


def clamp_t2(t1: float | None, t2: float | None) -> float | None:
    """Clamp T2 to the physical bound T2 <= 2*T1.

    Returns ``None`` unchanged so callers can decide how to treat missing data.
    """
    if t2 is None:
        return None
    if t1 is None or t1 <= 0.0:
        return t2
    return min(float(t2), T2_OVER_T1_LIMIT * float(t1))


def idle_infidelity(t: float, t1: float | None, t2: float | None) -> float:
    """Average infidelity accumulated by a qubit idling for ``t`` seconds.

    ``t2`` is clamped to ``2*T1`` internally, so callers may pass raw
    calibration values.  Missing/non-positive coherence times are treated as
    the fully-decohered limit (0.5) for any positive idle time.
    """
    if t is None or t <= 0.0:
        return 0.0
    t2c = clamp_t2(t1, t2)

    t1_bad = t1 is None or t1 <= _MIN_COHERENCE_S
    t2_bad = t2c is None or t2c <= _MIN_COHERENCE_S
    if t1_bad and t2_bad:
        return 0.5

    # math.exp(-x) underflows gracefully to 0.0 for large x, which is exactly
    # the fully-decohered limit we want.
    amp = 0.0 if t1_bad else math.exp(-float(t) / float(t1))
    deph = 0.0 if t2_bad else math.exp(-float(t) / float(t2c))

    eps = (3.0 - 2.0 * deph - amp) / 6.0
    # Guard against float dust on either side of the physical range.
    return min(0.5, max(0.0, eps))


def survival_probability(t: float, t1: float | None) -> float:
    """Probability an excited state survives ``t`` seconds of amplitude damping.

    Used only for reporting/intuition, not for ESP.
    """
    if t is None or t <= 0.0:
        return 1.0
    if t1 is None or t1 <= _MIN_COHERENCE_S:
        return 0.0
    return math.exp(-float(t) / float(t1))


def esp_from_terms(
    gate_errors: Iterable[float],
    idle_windows: Iterable[tuple[float, float | None, float | None]],
) -> float:
    """Estimated Success Probability.

    ``gate_errors``  -- per-instruction error rates (1q, 2q, readout).
    ``idle_windows`` -- ``(duration_s, t1, t2)`` triples, one per idle gap.

    ESP = prod(1 - eps_gate) * prod(1 - eps_idle).  Computed in log space so a
    few thousand terms do not underflow to zero.
    """
    log_esp = 0.0
    for err in gate_errors:
        e = 0.0 if err is None else float(err)
        e = min(1.0, max(0.0, e))
        if e >= 1.0:
            return 0.0
        log_esp += math.log1p(-e)
    for t, t1, t2 in idle_windows:
        e = idle_infidelity(t, t1, t2)
        if e >= 1.0:
            return 0.0
        log_esp += math.log1p(-e)
    if log_esp < -700.0:  # exp underflow threshold for float64
        return 0.0
    return math.exp(log_esp)


def esp_to_log10(esp: float) -> float:
    """log10(ESP), floored so a zero ESP does not produce -inf in reports."""
    if esp <= 0.0:
        return -float("inf")
    return math.log10(esp)


def hellinger_fidelity(p: dict[str, float], q: dict[str, float]) -> float:
    """Hellinger fidelity between two count/probability dicts.

    ``F = (sum_i sqrt(p_i q_i))**2``.  Inputs are normalised first, so raw shot
    counts are accepted.  Matches Qiskit's ``hellinger_fidelity`` definition.
    """
    p_tot = float(sum(p.values())) or 1.0
    q_tot = float(sum(q.values())) or 1.0
    keys = set(p) | set(q)
    bc = 0.0
    for k in keys:
        bc += math.sqrt((p.get(k, 0.0) / p_tot) * (q.get(k, 0.0) / q_tot))
    return min(1.0, bc * bc)


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile, no numpy dependency."""
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (len(ordered) - 1) * (pct / 100.0)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(ordered[int(idx)])
    return float(ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo))


def paired_mean_ci(
    deltas: Sequence[float], confidence: float = 0.95
) -> tuple[float, float, float]:
    """Mean and normal-approximation CI of a paired difference sample.

    Returns ``(mean, lo, hi)``.  A normal approximation is used rather than a
    t-distribution to keep this module free of scipy; with n >= 20 the
    difference is small, and the CI is reported as approximate.
    """
    n = len(deltas)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mean = sum(deltas) / n
    if n == 1:
        return (mean, mean, mean)
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    stderr = math.sqrt(var / n)
    z = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}.get(confidence, 1.9600)
    return (mean, mean - z * stderr, mean + z * stderr)
