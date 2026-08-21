"""Run the two transpilations side by side and measure both.

Fairness rules baked in here
----------------------------
1.  Both paths start from the *same* bound input circuit object.
2.  Both paths run the *same* preset pass manager at the same optimization
    level, over the *same* list of seeds.
3.  Both paths select their winner with the *same* objective (ESP).
    Giving the baseline one seed and the audited path twenty would manufacture
    a win out of nothing but extra sampling.

A caveat we state rather than hide: selecting by ESP means the headline ESP
delta is partly self-fulfilling, because ESP is also what the injected map
optimises.  The independent check is the Aer Hellinger fidelity in
``benchmarks/`` -- ESP is the model, Hellinger is the measurement.

The two audited strategies
--------------------------
``vf2_pinned``
    When VF2Layout (fed our T1/T2 map) finds an *exact* subgraph embedding, we
    pin it as ``initial_layout``.  Pinning also removes every VF2PostLayout
    from the preset pass manager, so nothing downstream can quietly overwrite
    the choice with Qiskit's own error map.  Zero routing overhead by
    construction -- this is the strategy that wins on sparse circuits.

``post_layout_relocate``
    Take the baseline's routed circuit and *relocate* it with
    ``VF2PostLayout(strict_direction=False)`` under our map.  The gate
    structure, the SWAPs and the depth are untouched; only the physical qubits
    change.  This is what wins on dense circuits, where the routing SABRE found
    is worth far more than any layout we could impose on it.

An earlier design pinned the post-layout result as an ``initial_layout`` and
re-transpiled.  That was measurably wrong: re-routing from a permutation chosen
for the *routed* interaction graph cost more SWAPs than the better qubits
saved (-22% ESP on a 7-qubit QFT mirror circuit, confirmed by Aer).  Relocating
the already-routed circuit is the correct use of VF2PostLayout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qiskit import QuantumCircuit
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
from .feature_extract import CircuitFeatures
from .metrics import SwapCounter, compute_metrics
from .models import CalibrationSnapshot, MetricSet
from .passes import ErrorMapBuild, InjectDecoherenceErrorMap, build_decoherence_error_map
from .progress import NULL_REPORTER, ProgressReporter
from .scheduling import ScheduleResult

# VF2 search budget.  Unbounded search on a 127-qubit device with a dense
# interaction graph can run for minutes, so the search is capped -- but only by
# limits that are a deterministic function of the input.
#
# `time_limit` is deliberately None. It is a *wall-clock* budget, so a machine
# under load stops the search at a different point and the audit returns a
# different layout for the same circuit, seed and calibration. This was not
# hypothetical: running two benchmark processes concurrently made su2_6x3
# disagree with itself on 1 of 20 seeds. `call_limit` and `max_trials` bound the
# same search by work done rather than time elapsed, which is reproducible.
VF2_CALL_LIMIT = 3_000_000
VF2_TIME_LIMIT = None
VF2_MAX_TRIALS = 25_000

STRATEGY_PINNED = "vf2_pinned"
STRATEGY_RELOCATE = "post_layout_relocate"


@dataclass
class AuditSettings:
    backend: str = "fake_sherbrooke"
    alpha: float = 1.0
    seeds: int = 20
    base_seed: int = 42
    optimization_level: int = 3
    schedule_method: str = "asap"
    post_layout_refine: bool = True
    # "normalized" corrects for VF2 exponentiating the diagonal by the
    # per-qubit 1q op count; "raw" is the literal design-doc formula.
    map_scaling: str = "normalized"
    # Benchmarks need every seed's circuit to run a paired simulation; the
    # CLI does not, and 20 x 127-qubit circuits is memory we can skip there.
    keep_circuits: bool = False

    def seed_list(self) -> list[int]:
        if self.seeds < 1:
            raise TranspileAuditError("--seeds must be >= 1")
        return [self.base_seed + i for i in range(self.seeds)]


@dataclass
class PathResult:
    """One transpilation strategy, evaluated over the whole seed list."""

    label: str
    best: MetricSet
    best_circuit: QuantumCircuit
    best_schedule: ScheduleResult
    all_metrics: list[MetricSet] = field(default_factory=list)
    all_circuits: list[QuantumCircuit] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def esp_series(self) -> list[float]:
        return [m.esp for m in self.all_metrics]


@dataclass
class AuditResult:
    baseline: PathResult
    baseline_default: MetricSet
    audited: PathResult
    audited_strategy: str
    strategies: dict[str, PathResult]
    error_map: ErrorMapBuild
    layout_info: dict
    settings: AuditSettings
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def recommendation(self) -> str:
        return "adopt_audited" if self.audited.best.esp > self.baseline.best.esp else "keep_baseline"


class _SeedTranspileLayout(AnalysisPass):
    """Replay a finished circuit's TranspileLayout into a fresh property set.

    ``ApplyLayout`` needs ``layout``/``final_layout``/``original_qubit_indices``
    to compose a post-layout onto an already-routed circuit; a standalone pass
    manager starts with none of them.
    """

    def __init__(self, transpile_layout):
        super().__init__()
        self._layout = transpile_layout

    def __hash__(self):  # Qiskit hashes every pass __init__ argument.
        return id(self)

    def run(self, dag):
        if self._layout is not None:
            self._layout.write_into_property_set(self.property_set)
        return dag


def _direction_fix_passes(target) -> list:
    """Repair ECR/CX direction after a relocation.

    ``VF2PostLayout(strict_direction=False)`` is the only branch that reads our
    injected map, and it is free to land a 2q gate on the uncalibrated
    orientation of a coupler.  The circuit is then not ISA-valid -- it fails at
    scheduling with "Duration of ecr on qubits [...] is not found" -- so the
    same direction fix-up the preset translation stage runs has to run here.
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
# Audited strategy 1: pin a VF2 layout chosen under the injected map
# --------------------------------------------------------------------------


def _vf2_layout_from_injection(
    unrolled: QuantumCircuit,
    target,
    inject: InjectDecoherenceErrorMap,
    seed: int,
) -> tuple[list[int] | None, str]:
    """Run ``Inject -> VF2Layout`` and read the chosen physical qubits out."""
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
            "Injected error map vanished from the property set.",
            hint="Qiskit's VF2 property-set contract has changed; see "
            "tests/test_injection_contract.py.",
        )
    reason = str(pm.property_set.get("VF2Layout_stop_reason", "unknown"))
    layout = pm.property_set.get("layout")
    if layout is None:
        return None, reason
    return [layout[q] for q in unrolled.qubits], reason


# --------------------------------------------------------------------------
# Audited strategy 2: relocate an already-routed circuit
# --------------------------------------------------------------------------


def _found_post_layout(property_set) -> bool:
    return property_set["post_layout"] is not None


def _relocate(
    routed: QuantumCircuit,
    target,
    inject: InjectDecoherenceErrorMap | None,
    seed: int,
) -> tuple[QuantumCircuit, str, bool]:
    """Move a routed circuit onto better physical qubits, gates untouched.

    ApplyLayout is gated behind a conditional controller rather than being run
    from a second pass manager: VF2PostLayout on a 127-qubit target is the
    expensive step here, and running it twice per seed doubled the cost of the
    whole relocate sweep for nothing.
    """
    passes = [_SeedTranspileLayout(getattr(routed, "layout", None))]
    if inject is not None:
        passes.append(inject)
    # inject=None leaves `vf2_avg_error_map` unset, so VF2PostLayout builds
    # Qiskit's own average map from the target. That is the control arm: same
    # relocation machinery, no T1/T2 knowledge. See benchmarks/bench.py --control.
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
# Sweeps
# --------------------------------------------------------------------------


def _best_of(
    metrics: list[MetricSet],
    circuits: list[QuantumCircuit],
    schedules: list[ScheduleResult],
) -> tuple[MetricSet, QuantumCircuit, ScheduleResult]:
    index = max(range(len(metrics)), key=lambda i: metrics[i].esp)
    return metrics[index], circuits[index], schedules[index]


def _run_baseline_sweep(
    circuit: QuantumCircuit,
    backend,
    target,
    snapshot: CalibrationSnapshot,
    settings: AuditSettings,
    *,
    reporter: ProgressReporter,
) -> tuple[PathResult, list[QuantumCircuit], list[int]]:
    metrics: list[MetricSet] = []
    circuits: list[QuantumCircuit] = []
    schedules: list[ScheduleResult] = []
    swaps: list[int] = []
    for seed in settings.seed_list():
        pm = generate_preset_pass_manager(
            optimization_level=settings.optimization_level,
            backend=backend,
            seed_transpiler=seed,
        )
        counter = SwapCounter()
        transpiled = pm.run(circuit, callback=counter)
        measured, schedule = compute_metrics(
            transpiled, snapshot, target,
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


def _run_pinned_sweep(
    circuit: QuantumCircuit,
    backend,
    target,
    snapshot: CalibrationSnapshot,
    settings: AuditSettings,
    layout: list[int],
    *,
    reporter: ProgressReporter,
) -> PathResult:
    metrics: list[MetricSet] = []
    circuits: list[QuantumCircuit] = []
    schedules: list[ScheduleResult] = []
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
            transpiled, snapshot, target,
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
    baseline_circuits: list[QuantumCircuit],
    baseline_swaps: list[int],
    target,
    snapshot: CalibrationSnapshot,
    settings: AuditSettings,
    inject: InjectDecoherenceErrorMap | None,
    *,
    reporter: ProgressReporter,
    label: str = STRATEGY_RELOCATE,
) -> PathResult:
    metrics: list[MetricSet] = []
    circuits: list[QuantumCircuit] = []
    schedules: list[ScheduleResult] = []
    moved = 0
    reasons: dict[str, int] = {}
    for seed, routed, swaps in zip(settings.seed_list(), baseline_circuits, baseline_swaps):
        relocated, reason, did_move = _relocate(routed, target, inject, seed)
        reasons[reason] = reasons.get(reason, 0) + 1
        moved += int(did_move)
        measured, schedule = compute_metrics(
            relocated, snapshot, target,
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
# Entry point
# --------------------------------------------------------------------------


def run_audit(
    bound_circuit: QuantumCircuit,
    unrolled: QuantumCircuit,
    features: CircuitFeatures,
    snapshot: CalibrationSnapshot,
    backend,
    settings: AuditSettings,
    *,
    reporter: ProgressReporter = NULL_REPORTER,
) -> AuditResult:
    target = backend.target
    warnings: list[str] = []
    notes: list[str] = []

    if features.num_qubits > snapshot.num_qubits:
        raise TranspileAuditError(
            f"Circuit needs {features.num_qubits} qubits but "
            f"{snapshot.backend_name} has {snapshot.num_qubits}."
        )

    build = build_decoherence_error_map(
        snapshot,
        t_est_s=features.t_est_s,
        alpha=settings.alpha,
        op_stats=features.op_stats,
        scaling=settings.map_scaling,
    )
    inject = InjectDecoherenceErrorMap(
        snapshot, t_est_s=features.t_est_s, alpha=settings.alpha, build=build
    )
    reporter.step(
        "error_map_built",
        t_est_s=features.t_est_s,
        alpha=settings.alpha,
        scaling=build.scaling,
        worst=build.worst_qubits(3),
    )

    # ---- baseline (also the substrate for the relocate strategy) ----------
    reporter.step("sweep_start", path="baseline", seeds=settings.seeds)
    baseline, baseline_circuits, baseline_swaps = _run_baseline_sweep(
        bound_circuit, backend, target, snapshot, settings, reporter=reporter
    )

    strategies: dict[str, PathResult] = {}
    layout_info: dict = {}

    # ---- strategy 1: pinned VF2 layout -----------------------------------
    layout, reason = _vf2_layout_from_injection(unrolled, target, inject, settings.base_seed)
    layout_info["vf2_stop_reason"] = reason
    layout_info["vf2_layout"] = layout
    if layout is not None:
        reporter.step("vf2_layout", layout=layout, reason=reason)
        reporter.step("sweep_start", path=STRATEGY_PINNED, seeds=settings.seeds)
        strategies[STRATEGY_PINNED] = _run_pinned_sweep(
            bound_circuit, backend, target, snapshot, settings, layout, reporter=reporter
        )
    else:
        notes.append(
            "VF2Layout found no exact subgraph embedding for this interaction graph "
            f"({reason}), so no zero-SWAP layout exists: the audit falls back to "
            "relocating SABRE's routed circuit onto better qubits."
        )
        reporter.step("vf2_layout_none", reason=reason)

    # ---- strategy 2: relocate the routed circuit -------------------------
    if settings.post_layout_refine:
        reporter.step("sweep_start", path=STRATEGY_RELOCATE, seeds=settings.seeds)
        strategies[STRATEGY_RELOCATE] = _run_relocate_sweep(
            baseline_circuits, baseline_swaps, target, snapshot, settings, inject,
            reporter=reporter,
        )

    if not strategies:
        raise TranspileAuditError(
            "No audited strategy was available: VF2Layout found nothing and "
            "post-layout relocation was disabled.",
            hint="Drop --no-post-layout, or audit a circuit whose interaction "
            "graph embeds in the device.",
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
            "The audited layout did not beat the baseline on ESP for this circuit. "
            "That is a real result, not a failure: on a well-calibrated device a "
            "compact SABRE layout can already sit on good qubits. Recommendation: "
            "keep the baseline transpilation."
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


def run_control_relocation(
    baseline_circuits: list[QuantumCircuit],
    baseline_swaps: list[int],
    target,
    snapshot: CalibrationSnapshot,
    settings: AuditSettings,
    *,
    reporter: ProgressReporter = NULL_REPORTER,
) -> PathResult:
    """Control arm: relocate with **Qiskit's own** error map, no injection.

    Isolates how much of the measured gain comes from running a non-strict
    ``VF2PostLayout`` at all (the preset only ever runs a *strict* one in its
    optimization stage, which searches a far smaller space) versus how much
    comes from the T1/T2 information we inject. Without this control the two
    effects are confounded.
    """
    return _run_relocate_sweep(
        baseline_circuits, baseline_swaps, target, snapshot, settings, None,
        reporter=reporter, label="relocate_default_map",
    )


def run_baseline_only(
    bound_circuit: QuantumCircuit,
    backend,
    snapshot: CalibrationSnapshot,
    settings: AuditSettings,
    *,
    reporter: ProgressReporter = NULL_REPORTER,
) -> tuple[PathResult, list[QuantumCircuit], list[int]]:
    """Baseline sweep on its own, for harnesses that need the routed circuits."""
    return _run_baseline_sweep(
        bound_circuit, backend, backend.target, snapshot, settings, reporter=reporter
    )
