"""run the two transpilations side by side and measure both.

fairness rules, because otherwise the comparison is worthless:
both paths start from the same bound circuit object, both run the same preset
pass manager at the same optimization level over the same seed list, and both
pick their winner with the same objective (esp). giving the baseline one seed
and the audited path twenty would manufacture a win out of nothing.

caveat i'm not going to hide: selecting by esp makes the headline esp delta
partly self fulfilling, since esp is also what the injected map optimises. the
independent check is the aer hellinger fidelity in benchmarks/. esp is the
model, hellinger is the measurement.

there are two audited strategies and which one wins depends on the circuit.
dense circuits need the relocate thing, sparse ones can just pin the layout:

vf2_pinned, when VF2Layout (fed our map) finds an exact embedding we pin it as
initial_layout. pinning also strips every VF2PostLayout out of the preset, so
nothing downstream can overwrite our choice with qiskit's own error map. zero
routing overhead by construction.

post_layout_relocate, take the baseline's already routed circuit and move it
onto better qubits with VF2PostLayout(strict_direction=False). the gates, the
swaps and the depth all stay the same, only the physical qubits change.

first version of this pinned the post layout result as an initial_layout and
re-transpiled. that was just wrong: re-routing from a permutation chosen for
the *routed* interaction graph cost more swaps than the better qubits saved
(-22% esp on qft mirror 7, aer agreed). relocating the routed circuit is the
right way to use VF2PostLayout
"""

import json
import sys
import time
from dataclasses import dataclass, field

from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as _SEL
from qiskit.passmanager.flow_controllers import ConditionalController
from qiskit.transpiler import PassManager, generate_preset_pass_manager
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.passes import (
    ApplyLayout,
    BasisTranslator,
    CheckGateDirection,
    GateDirection,
    Optimize1qGatesDecomposition,
    VF2Layout,
    VF2PostLayout,
)

from .errors import InjectionContractError, TranspileAuditError
from .metrics import SwapCounter, compute_metrics
from .passes import InjectDecoherenceErrorMap, build_decoherence_error_map

# how much vf2 search we allow. unbounded search on 127 qubits with a dense
# interaction graph can run for minutes.
#
# time_limit stays None on purpose. it's a wall clock budget, so a loaded
# machine stops the search somewhere else and you get a different layout for
# the same circuit and seed. not hypothetical, running two benchmarks at once
# made su2_6x3 disagree with itself on 1 seed out of 20. call_limit and
# max_trials bound the same search by work done, which is reproducible
VF2_CALL_LIMIT = 3_000_000
VF2_TIME_LIMIT = None
VF2_MAX_TRIALS = 25_000

STRATEGY_PINNED = "vf2_pinned"
STRATEGY_RELOCATE = "post_layout_relocate"


# --------------------------------------------------------------------------
# progress events, ndjson on stderr
# --------------------------------------------------------------------------
# stdout belongs to the report (and to exactly one json document with --json),
# so anything chatty goes to stderr instead, one json object per line


@dataclass
class ProgressReporter:
    enabled: bool = True
    quiet: bool = False
    _t0: float = field(default_factory=time.monotonic)

    def emit(self, event, **fields):
        if not self.enabled or self.quiet:
            return
        payload = {"ts": round(time.monotonic() - self._t0, 4), "event": event, **fields}
        try:
            sys.stderr.write(json.dumps(payload, default=str) + "\n")
            sys.stderr.flush()
        except (OSError, ValueError):
            pass

    def step(self, name, **fields):
        self.emit("step", step=name, **fields)

    def warn(self, message, **fields):
        self.emit("warning", message=message, **fields)

    def error(self, message, **fields):
        self.emit("error", message=message, **fields)


NULL_REPORTER = ProgressReporter(enabled=False)


@dataclass
class AuditSettings:
    backend: str = "fake_sherbrooke"
    alpha: float = 1.0
    seeds: int = 20
    base_seed: int = 42
    optimization_level: int = 3
    schedule_method: str = "asap"
    post_layout_refine: bool = True
    # normalized corrects for vf2 exponentiating the diagonal, raw is the
    # literal formula
    map_scaling: str = "normalized"
    # benchmarks need every seed's circuit to run paired simulations, the cli
    # doesn't and 20 x 127 qubit circuits is a lot of memory to hold for nothing
    keep_circuits: bool = False

    def seed_list(self):
        if self.seeds < 1:
            raise TranspileAuditError("--seeds must be >= 1")
        return [self.base_seed + i for i in range(self.seeds)]


@dataclass
class PathResult:
    """one strategy, evaluated over the whole seed list"""

    label: str
    best: object
    best_circuit: object
    best_schedule: object
    all_metrics: list = field(default_factory=list)
    all_circuits: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def esp_series(self):
        return [m.esp for m in self.all_metrics]


@dataclass
class AuditResult:
    baseline: PathResult
    baseline_default: object
    audited: PathResult
    audited_strategy: str
    strategies: dict
    error_map: object
    layout_info: dict
    settings: AuditSettings
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def recommendation(self):
        if self.audited.best.esp > self.baseline.best.esp:
            return "adopt_audited"
        return "keep_baseline"


class _SeedTranspileLayout(AnalysisPass):
    """replay a finished circuit's TranspileLayout into a fresh property set.

    ApplyLayout needs layout/final_layout/original_qubit_indices to compose a
    post layout onto an already routed circuit, and a standalone pass manager
    starts with none of those
    """

    def __init__(self, transpile_layout):
        super().__init__()
        self._layout = transpile_layout

    def __hash__(self):  # qiskit hashes every pass argument
        return id(self)

    def run(self, dag):
        if self._layout is not None:
            self._layout.write_into_property_set(self.property_set)
        return dag


def _direction_fix_passes(target):
    """fix ecr/cx direction after a relocation.

    VF2PostLayout(strict_direction=False) is the only branch that reads our map
    and it's free to put a 2q gate on the uncalibrated orientation of a coupler.
    the circuit is then not ISA valid and blows up at scheduling with "Duration
    of ecr on qubits [...] is not found", so we run the same direction fixup the
    preset translation stage does
    """
    basis = list(target.operation_names)

    def _needs_fix(property_set):
        return not property_set["is_direction_mapped"]

    return [
        CheckGateDirection(None, target=target),
        ConditionalController(
            [
                GateDirection(None, target=target),
                BasisTranslator(_SEL, basis, target),
                Optimize1qGatesDecomposition(target=target),
            ],
            condition=_needs_fix,
        ),
    ]


# --------------------------------------------------------------------------
# strategy 1: pin a vf2 layout chosen under our map
# --------------------------------------------------------------------------


def _vf2_layout_from_injection(unrolled, target, inject, seed):
    """run inject -> VF2Layout and read the physical qubits back out"""
    pm = PassManager(
        [
            inject,
            VF2Layout(
                target=target,
                seed=seed,
                call_limit=VF2_CALL_LIMIT,
                time_limit=VF2_TIME_LIMIT,
                max_trials=VF2_MAX_TRIALS,
                strict_direction=False,
            ),
        ]
    )
    pm.run(unrolled)
    if pm.property_set.get("vf2_avg_error_map") is None:
        raise InjectionContractError(
            "our error map vanished out of the property set",
            hint="qiskit's vf2 property set contract changed, see "
            "tests/test_injection_contract.py",
        )
    reason = str(pm.property_set.get("VF2Layout_stop_reason", "unknown"))
    layout = pm.property_set.get("layout")
    if layout is None:
        return None, reason
    return [layout[q] for q in unrolled.qubits], reason


# --------------------------------------------------------------------------
# strategy 2: relocate an already routed circuit
# --------------------------------------------------------------------------


def _found_post_layout(property_set):
    return property_set["post_layout"] is not None


def _relocate(routed, target, inject, seed):
    """move a routed circuit onto better qubits, gates untouched.

    ApplyLayout sits behind a conditional controller rather than in a second
    pass manager becuase VF2PostLayout on 127 qubits is the expensive step and
    running it twice per seed doubled the cost of the whole sweep for nothing
    """
    passes = [_SeedTranspileLayout(getattr(routed, "layout", None))]
    if inject is not None:
        passes.append(inject)
    # inject=None leaves vf2_avg_error_map unset so VF2PostLayout builds
    # qiskit's own average map from the target. that's the control arm: same
    # machinery, no T1/T2. see benchmarks/bench.py --control
    passes += [
        VF2PostLayout(
            target=target,
            seed=seed,
            strict_direction=False,
            call_limit=VF2_CALL_LIMIT,
            time_limit=VF2_TIME_LIMIT,
            max_trials=VF2_MAX_TRIALS,
        ),
        ConditionalController(
            [ApplyLayout(), *_direction_fix_passes(target)],
            condition=_found_post_layout,
        ),
    ]
    pm = PassManager(passes)
    out = pm.run(routed)
    reason = str(pm.property_set.get("VF2PostLayout_stop_reason", "unknown"))
    moved = "SOLUTION_FOUND" in reason and "NO_" not in reason
    return out, reason, moved


# --------------------------------------------------------------------------
# the seed sweeps
# --------------------------------------------------------------------------


def _best_of(metrics, circuits, schedules):
    index = max(range(len(metrics)), key=lambda i: metrics[i].esp)
    return metrics[index], circuits[index], schedules[index]


def _run_baseline_sweep(circuit, backend, target, snap, settings, *, reporter):
    metrics = []
    circuits = []
    schedules = []
    swaps = []
    for seed in settings.seed_list():
        pm = generate_preset_pass_manager(
            optimization_level=settings.optimization_level,
            backend=backend,
            seed_transpiler=seed,
        )
        counter = SwapCounter()
        transpiled = pm.run(circuit, callback=counter)
        measured, schedule = compute_metrics(
            transpiled, snap, target,
            label="baseline", seed=seed, swaps=counter.added,
            schedule_method=settings.schedule_method,
        )
        metrics.append(measured)
        circuits.append(transpiled)
        schedules.append(schedule)
        swaps.append(counter.added)
        reporter.step(
            "transpile", path="baseline", seed=seed,
            esp=round(measured.esp, 6), swaps=counter.added,
            two_q=measured.two_qubit_gates,
        )
    best, best_circuit, best_schedule = _best_of(metrics, circuits, schedules)
    path = PathResult(
        label="baseline",
        best=best,
        best_circuit=best_circuit,
        best_schedule=best_schedule,
        all_metrics=metrics,
        all_circuits=circuits if settings.keep_circuits else [],
    )
    return path, circuits, swaps


def _run_pinned_sweep(circuit, backend, target, snap, settings, layout, *, reporter):
    metrics = []
    circuits = []
    schedules = []
    for seed in settings.seed_list():
        pm = generate_preset_pass_manager(
            optimization_level=settings.optimization_level,
            backend=backend,
            seed_transpiler=seed,
            initial_layout=list(layout),
        )
        counter = SwapCounter()
        transpiled = pm.run(circuit, callback=counter)
        measured, schedule = compute_metrics(
            transpiled, snap, target,
            label=STRATEGY_PINNED, seed=seed, swaps=counter.added,
            schedule_method=settings.schedule_method,
        )
        metrics.append(measured)
        circuits.append(transpiled)
        schedules.append(schedule)
        reporter.step(
            "transpile", path=STRATEGY_PINNED, seed=seed,
            esp=round(measured.esp, 6), swaps=counter.added,
        )
    best, best_circuit, best_schedule = _best_of(metrics, circuits, schedules)
    return PathResult(
        label=STRATEGY_PINNED,
        best=best,
        best_circuit=best_circuit,
        best_schedule=best_schedule,
        all_metrics=metrics,
        all_circuits=circuits if settings.keep_circuits else [],
        info={"initial_layout": list(layout)},
    )


def _run_relocate_sweep(
    baseline_circuits, baseline_swaps, target, snap, settings, inject,
    *, reporter, label=STRATEGY_RELOCATE,
):
    metrics = []
    circuits = []
    schedules = []
    moved = 0
    reasons = {}
    for seed, routed, swaps in zip(settings.seed_list(), baseline_circuits, baseline_swaps):
        relocated, reason, did_move = _relocate(routed, target, inject, seed)
        reasons[reason] = reasons.get(reason, 0) + 1
        moved += int(did_move)
        measured, schedule = compute_metrics(
            relocated, snap, target,
            label=label, seed=seed, swaps=swaps,
            schedule_method=settings.schedule_method,
        )
        metrics.append(measured)
        circuits.append(relocated)
        schedules.append(schedule)
        reporter.step(
            "relocate", path=label, seed=seed,
            esp=round(measured.esp, 6), moved=did_move,
        )
    best, best_circuit, best_schedule = _best_of(metrics, circuits, schedules)
    return PathResult(
        label=label,
        best=best,
        best_circuit=best_circuit,
        best_schedule=best_schedule,
        all_metrics=metrics,
        all_circuits=circuits if settings.keep_circuits else [],
        info={"relocated_seeds": moved, "stop_reasons": reasons},
    )


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def run_audit(bound_circuit, unrolled, features, snap, backend, settings,
              *, reporter=NULL_REPORTER):
    target = backend.target
    warnings = []
    notes = []

    if features.num_qubits > snap.num_qubits:
        raise TranspileAuditError(
            f"circuit needs {features.num_qubits} qubits but "
            f"{snap.backend_name} only has {snap.num_qubits}"
        )

    build = build_decoherence_error_map(
        snap,
        t_est_s=features.t_est_s,
        alpha=settings.alpha,
        op_stats=features.op_stats,
        scaling=settings.map_scaling,
    )
    inject = InjectDecoherenceErrorMap(
        snap, t_est_s=features.t_est_s, alpha=settings.alpha, build=build
    )
    reporter.step(
        "error_map_built",
        t_est_s=features.t_est_s,
        alpha=settings.alpha,
        scaling=build.scaling,
        worst=build.worst_qubits(3),
    )

    # baseline first, the relocate strategy builds on top of its circuits
    reporter.step("sweep_start", path="baseline", seeds=settings.seeds)
    baseline, baseline_circuits, baseline_swaps = _run_baseline_sweep(
        bound_circuit, backend, target, snap, settings, reporter=reporter
    )

    strategies = {}
    layout_info = {}

    layout, reason = _vf2_layout_from_injection(unrolled, target, inject, settings.base_seed)
    layout_info["vf2_stop_reason"] = reason
    layout_info["vf2_layout"] = layout
    if layout is not None:
        reporter.step("vf2_layout", layout=layout, reason=reason)
        reporter.step("sweep_start", path=STRATEGY_PINNED, seeds=settings.seeds)
        strategies[STRATEGY_PINNED] = _run_pinned_sweep(
            bound_circuit, backend, target, snap, settings, layout, reporter=reporter
        )
    else:
        notes.append(
            "VF2Layout found no exact subgraph embedding for this interaction graph "
            f"({reason}), so there's no zero-swap layout to pin. falling back to "
            "relocating sabre's routed circuit onto better qubits."
        )
        reporter.step("vf2_layout_none", reason=reason)

    if settings.post_layout_refine:
        reporter.step("sweep_start", path=STRATEGY_RELOCATE, seeds=settings.seeds)
        strategies[STRATEGY_RELOCATE] = _run_relocate_sweep(
            baseline_circuits, baseline_swaps, target, snap, settings, inject,
            reporter=reporter,
        )

    if not strategies:
        raise TranspileAuditError(
            "no audited strategy left: vf2 found nothing and post layout relocation "
            "was turned off",
            hint="drop --no-post-layout, or audit a circuit whose interaction graph "
            "actually embeds in the device",
        )

    audited_strategy = max(strategies, key=lambda k: strategies[k].best.esp)
    audited = strategies[audited_strategy]
    layout_info["strategy"] = audited_strategy
    layout_info["strategy_esp"] = {k: v.best.esp for k, v in strategies.items()}
    layout_info["final_layout"] = list(audited.best.layout)
    layout_info["source"] = audited_strategy

    layout_info.update(audited.info)

    baseline_default = next(
        (m for m in baseline.all_metrics if m.seed == settings.base_seed),
        baseline.all_metrics[0],
    )

    if audited.best.esp <= baseline.best.esp:
        notes.append(
            "the audited layout didn't beat the baseline on esp here. that's a real "
            "result and not a failure, on a well calibrated device a compact sabre "
            "layout can already be sitting on good qubits. keep the baseline."
        )

    return AuditResult(
        baseline=baseline,
        baseline_default=baseline_default,
        audited=audited,
        audited_strategy=audited_strategy,
        strategies=strategies,
        error_map=build,
        layout_info=layout_info,
        settings=settings,
        warnings=warnings,
        notes=notes,
    )


def run_control_relocation(baseline_circuits, baseline_swaps, target, snap, settings,
                           *, reporter=NULL_REPORTER):
    """control arm: same relocation but with qiskit's own error map.

    tells us how much of the gain is just "we ran a non-strict VF2PostLayout at
    all" (the preset only ever runs a strict one, which searches a way smaller
    space) versus how much is the T1/T2 we inject. without this the two effects
    are confounded
    """
    return _run_relocate_sweep(
        baseline_circuits, baseline_swaps, target, snap, settings, None,
        reporter=reporter, label="relocate_default_map",
    )


def run_baseline_only(bound_circuit, backend, snap, settings, *, reporter=NULL_REPORTER):
    """just the baseline sweep, for harnesses that want the routed circuits"""
    return _run_baseline_sweep(
        bound_circuit, backend, backend.target, snap, settings, reporter=reporter
    )
