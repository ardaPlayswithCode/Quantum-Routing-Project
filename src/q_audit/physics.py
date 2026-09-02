"""all the noise math. no qiskit in here on purpose so it's easy to unit test.

the idle formula is

    eps_idle(t) = (3 - 2*exp(-t/T2) - exp(-t/T1)) / 6

which is the average infidelity of a thermal relaxation channel with no thermal
population. eps(0) = 0 and eps(inf) = 1/2, both of which the tests check.

only idle gaps get charged decoherence. the gate errors from calibration
already include whatever decohered during the gate, so adding relaxation on top
of a gate error would count it twice.
"""

import math

# T2 > 2*T1 is unphysical. hardware reports it sometimes anyway becuase T1 and
# T2 come from two different experiments
T2_OVER_T1_LIMIT = 2.0

# below this just call it dead instead of dividing by ~0
_MIN_COHERENCE_S = 1e-12


def clamp_t2(t1, t2):
    """clamp T2 to 2*T1. None stays None so callers can decide what to do"""
    if t2 is None:
        return None
    if t1 is None or t1 <= 0.0:
        return t2
    return min(float(t2), T2_OVER_T1_LIMIT * float(t1))


def idle_infidelity(t, t1, t2) -> float:
    """infidelity a qubit picks up sitting idle for t seconds.

    clamps t2 itself so you can pass raw calibration values. missing coherence
    times count as fully decohered
    """
    if t is None or t <= 0.0:
        return 0.0
    t2c = clamp_t2(t1, t2)

    t1_bad = t1 is None or t1 <= _MIN_COHERENCE_S
    t2_bad = t2c is None or t2c <= _MIN_COHERENCE_S
    if t1_bad and t2_bad:
        return 0.5

    # exp(-big) underflows to 0.0 which is exactly the fully decohered limit,
    # so no special case needed for long idles
    amp = 0.0 if t1_bad else math.exp(-float(t) / float(t1))
    deph = 0.0 if t2_bad else math.exp(-float(t) / float(t2c))

    eps = (3.0 - 2.0 * deph - amp) / 6.0
    # float dust can push this a hair outside [0, 0.5]
    return min(0.5, max(0.0, eps))


def survival_probability(t, t1) -> float:
    """chance an excited state survives t seconds. only used for reporting"""
    if t is None or t <= 0.0:
        return 1.0
    if t1 is None or t1 <= _MIN_COHERENCE_S:
        return 0.0
    return math.exp(-float(t) / float(t1))


def esp_from_terms(gate_errors, idle_windows) -> float:
    """estimated success probability.

    gate_errors is a list of per instruction error rates, idle_windows is
    (duration, t1, t2) per idle gap. ESP = prod(1-eps). done in log space,
    a few thousand terms multiplied directly underflows to 0
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
    if log_esp < -700.0:  # exp() underflows past about here
        return 0.0
    return math.exp(log_esp)


def esp_to_log10(esp) -> float:
    if esp <= 0.0:
        return -float("inf")
    return math.log10(esp)


def hellinger_fidelity(p, q) -> float:
    """hellinger fidelity between two count dicts, F = (sum sqrt(p*q))^2.

    normalises first so you can pass raw shot counts. same definition qiskit uses
    """
    p_tot = float(sum(p.values())) or 1.0
    q_tot = float(sum(q.values())) or 1.0
    keys = set(p) | set(q)
    bc = 0.0
    for k in keys:
        bc += math.sqrt((p.get(k, 0.0) / p_tot) * (q.get(k, 0.0) / q_tot))
    return min(1.0, bc * bc)


def percentile(values, pct) -> float:
    """percentile with linear interpolation. didn't want to pull in numpy"""
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


def paired_mean_ci(deltas, confidence=0.95):
    """mean and CI of a paired difference sample, returns (mean, lo, hi).

    normal approximation instead of a t distribution so we don't need scipy.
    at n=20 the difference is tiny and we call the CI approximate anyway
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
