"""Orchestration: fairness of the paired comparison, layout validity, settings."""

from __future__ import annotations

import pytest

from q_audit.feature_extract import extract_features
from q_audit.transpile_audit import AuditSettings, run_audit

pytestmark = pytest.mark.slow

SEEDS = 3  # keep the suite quick; fairness properties do not depend on N


@pytest.fixture(scope="module")
def audit(request):
    snapshot = request.getfixturevalue("snapshot")
    backend = request.getfixturevalue("fake_backend")
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(7, name="ghz7")
    qc.h(0)
    for i in range(6):
        qc.cx(i, i + 1)
    qc.measure_all()
    bound, unrolled, features = extract_features(qc, snapshot)
    settings = AuditSettings(seeds=SEEDS)
    return run_audit(bound, unrolled, features, snapshot, backend, settings), features


def test_seed_lists_are_identical(audit):
    """The comparison is invalid the moment the two paths get different budgets."""
    result, _ = audit
    baseline_seeds = [m.seed for m in result.baseline.all_metrics]
    audited_seeds = [m.seed for m in result.audited.all_metrics]
    assert baseline_seeds == audited_seeds == result.settings.seed_list()
    assert len(baseline_seeds) == SEEDS


def test_both_paths_report_the_same_number_of_trials(audit):
    result, _ = audit
    assert len(result.baseline.esp_series()) == len(result.audited.esp_series())


def test_best_is_actually_the_best(audit):
    result, _ = audit
    for path in (result.baseline, result.audited):
        assert path.best.esp == max(m.esp for m in path.all_metrics)


def test_layouts_are_valid_physical_qubits(audit, snapshot):
    result, features = audit
    for metrics in (result.baseline.best, result.audited.best):
        assert len(metrics.layout) == features.num_qubits
        assert len(set(metrics.layout)) == len(metrics.layout)
        assert all(0 <= q < snapshot.num_qubits for q in metrics.layout)


def test_audited_layout_is_actually_used(audit):
    """The pinned layout must survive the pass manager, not be re-chosen."""
    result, _ = audit
    assert set(result.audited.best.layout) == set(result.layout_info["final_layout"])


def test_baseline_default_seed_is_recorded(audit):
    result, _ = audit
    assert result.baseline_default.seed == result.settings.base_seed


def test_error_map_is_reported(audit):
    result, _ = audit
    build = result.error_map
    assert build.t_est_s > 0
    assert len(build.diagonal) == 127
    assert all(0.0 <= v <= 1.0 for v in build.diagonal.values())
    assert build.scaling in ("raw", "normalized")


def test_layout_info_records_provenance(audit):
    from q_audit.transpile_audit import STRATEGY_PINNED, STRATEGY_RELOCATE

    result, _ = audit
    assert result.audited_strategy in (STRATEGY_PINNED, STRATEGY_RELOCATE)
    assert result.layout_info["source"] == result.audited_strategy
    assert "vf2_stop_reason" in result.layout_info
    assert set(result.layout_info["strategy_esp"]) == set(result.strategies)


def test_audited_is_the_best_available_strategy(audit):
    result, _ = audit
    assert result.audited.best.esp == max(
        path.best.esp for path in result.strategies.values()
    )


def test_relocation_preserves_routing_cost(audit):
    """The relocate strategy must not change gate structure, only placement."""
    from q_audit.transpile_audit import STRATEGY_RELOCATE

    result, _ = audit
    relocated = result.strategies.get(STRATEGY_RELOCATE)
    if relocated is None:
        pytest.skip("relocation disabled")
    for base, moved in zip(result.baseline.all_metrics, relocated.all_metrics):
        assert moved.swap_gates == base.swap_gates
        # Direction fix-up may add 1q gates around a flipped 2q gate, but the
        # two-qubit count is fixed by the routing we inherited.
        assert moved.two_qubit_gates == base.two_qubit_gates


def test_recommendation_matches_the_numbers(audit):
    result, _ = audit
    expected = (
        "adopt_audited"
        if result.audited.best.esp > result.baseline.best.esp
        else "keep_baseline"
    )
    assert result.recommendation == expected


def test_metrics_are_reproducible(snapshot, fake_backend, ghz7):
    """Same inputs, same seeds -> the same reported answer.

    Note what is *not* asserted: per-seed equality. Qiskit 2.5.1's preset pass
    manager is itself non-deterministic at optimization_level 2 and 3 even with
    a fixed ``seed_transpiler`` (see
    ``test_qiskit_preset_is_not_bit_reproducible_at_o3``), so an individual
    seed's circuit can differ between runs through no fault of ours. The
    user-visible headline -- best-of-N and the layout it picked -- is what must
    hold, and does.
    """
    bound, unrolled, features = extract_features(ghz7, snapshot)
    settings = AuditSettings(seeds=2)
    first = run_audit(bound, unrolled, features, snapshot, fake_backend, settings)
    second = run_audit(bound, unrolled, features, snapshot, fake_backend, settings)
    assert first.audited.best.esp == second.audited.best.esp
    assert first.audited.best.layout == second.audited.best.layout
    assert first.baseline.best.esp == second.baseline.best.esp


def test_raw_and_normalized_scaling_both_run(snapshot, fake_backend, ghz7):
    bound, unrolled, features = extract_features(ghz7, snapshot)
    results = {}
    for scaling in ("raw", "normalized"):
        result = run_audit(
            bound, unrolled, features, snapshot, fake_backend,
            AuditSettings(seeds=1, map_scaling=scaling),
        )
        results[scaling] = result
        assert result.error_map.scaling == scaling
    # They are different objectives, so they may disagree -- both must be valid.
    for result in results.values():
        assert 0.0 < result.audited.best.esp <= 1.0


def test_post_layout_refinement_can_be_disabled(snapshot, fake_backend, ghz7):
    from q_audit.transpile_audit import STRATEGY_PINNED, STRATEGY_RELOCATE

    bound, unrolled, features = extract_features(ghz7, snapshot)
    result = run_audit(
        bound, unrolled, features, snapshot, fake_backend,
        AuditSettings(seeds=1, post_layout_refine=False),
    )
    assert STRATEGY_RELOCATE not in result.strategies
    assert result.audited_strategy == STRATEGY_PINNED
    assert result.layout_info["final_layout"] == result.layout_info["vf2_layout"]


def test_relocate_only_when_vf2_cannot_embed(snapshot, fake_backend):
    """A dense circuit has no exact embedding; the relocate strategy carries it."""
    from qiskit import QuantumCircuit

    from q_audit.transpile_audit import STRATEGY_PINNED, STRATEGY_RELOCATE

    qc = QuantumCircuit(6)
    for i in range(6):
        for j in range(i + 1, 6):
            qc.cx(i, j)
    qc.measure_all()
    bound, unrolled, features = extract_features(qc, snapshot)
    result = run_audit(
        bound, unrolled, features, snapshot, fake_backend, AuditSettings(seeds=1)
    )
    assert STRATEGY_PINNED not in result.strategies
    assert result.audited_strategy == STRATEGY_RELOCATE
    assert any("no exact subgraph embedding" in n for n in result.notes)


def test_circuit_wider_than_device_is_rejected(snapshot, fake_backend):
    from qiskit import QuantumCircuit

    from q_audit.errors import TranspileAuditError

    qc = QuantumCircuit(200)
    qc.h(range(200))
    bound, unrolled, features = extract_features(qc, snapshot)
    with pytest.raises(TranspileAuditError):
        run_audit(
            bound, unrolled, features, snapshot, fake_backend, AuditSettings(seeds=1)
        )


@pytest.mark.aer
@pytest.mark.parametrize("builder", ["ghz", "dense"])
def test_audited_circuit_computes_the_same_thing(snapshot, fake_backend, builder):
    """THE load-bearing invariant.

    Relocating a routed circuit permutes physical qubits under it. If that
    permutation were not composed correctly into the layout -- or if the
    measure-to-clbit mapping did not follow -- every fidelity number this tool
    reports would be measuring the wrong circuit, and it would look like a
    *better* result rather than a broken one. So: simulate both paths with no
    noise at all and demand they reproduce the ideal distribution.
    """
    pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    from q_audit.physics import hellinger_fidelity
    from q_audit.verify import ensure_measured, ideal_distribution

    if builder == "ghz":
        qc = QuantumCircuit(7, name="ghz7")
        qc.h(0)
        for i in range(6):
            qc.cx(i, i + 1)
    else:  # forces routing, so the relocate strategy is exercised
        qc = QuantumCircuit(6, name="dense6")
        for i in range(6):
            for j in range(i + 1, 6):
                qc.cx(i, j)
    qc.measure_all()

    bound, unrolled, features = extract_features(qc, snapshot)
    result = run_audit(
        bound, unrolled, features, snapshot, fake_backend, AuditSettings(seeds=2)
    )
    ideal = ideal_distribution(ensure_measured(bound))
    simulator = AerSimulator()  # noiseless on purpose

    for path in (result.baseline, result.audited):
        counts = simulator.run(
            path.best_circuit, shots=20000, seed_simulator=11
        ).result().get_counts()
        fidelity = hellinger_fidelity(ideal, {k: float(v) for k, v in counts.items()})
        assert fidelity > 0.99, (
            f"{path.label} does not compute the input circuit "
            f"(noiseless fidelity {fidelity:.4f})"
        )


def test_qiskit_preset_is_not_bit_reproducible_at_o3(fake_backend):
    """Documents an upstream defect that bounds what q-audit can promise.

    ``generate_preset_pass_manager(optimization_level=3, seed_transpiler=42)``
    run repeatedly on the same circuit in the same process returns more than one
    distinct result on Qiskit 2.5.1 -- different layouts *and* different depths.
    optimization_level=1 does not do this, so it is something in the O2/O3
    optimization loop, not the layout or routing passes (each is deterministic
    in isolation).

    This test asserts only what is stable: every result is a valid ISA circuit
    on the same qubit count. It exists so that a future reader who sees q-audit
    report slightly different per-seed numbers between runs looks upstream
    instead of hunting a bug here. If Qiskit fixes this, the observation below
    collapses to a single variant and the test still passes.
    """
    from qiskit.transpiler import generate_preset_pass_manager

    from benchmarks.circuits import SUITE

    circuit = SUITE["su2_6x3"].build()
    seen = set()
    for _ in range(6):
        pm = generate_preset_pass_manager(
            optimization_level=3, backend=fake_backend, seed_transpiler=42
        )
        transpiled = pm.run(circuit)
        assert transpiled.num_qubits == fake_backend.num_qubits
        seen.add(
            (
                tuple(transpiled.layout.initial_index_layout(filter_ancillas=True)),
                transpiled.depth(),
            )
        )
    # Not an assertion about the count -- just a guard that nothing exploded.
    assert 1 <= len(seen) <= 6
