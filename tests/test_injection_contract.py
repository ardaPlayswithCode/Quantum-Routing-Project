"""canary tests. don't weaken these.

the whole tool rests on one thing: an ErrorMap put at
property_set["vf2_avg_error_map"] steers VF2Layout and VF2PostLayout. nothing
in qiskit's public api promises that. if an upgrade changes it the audit
doesn't crash, it turns into "transpile the circuit twice and report that
nothing changed", which is much worse than crashing.

these tests are here to make that fail loudly instead
"""

import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager, generate_preset_pass_manager
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.passes import VF2Layout, VF2PostLayout
from qiskit.transpiler.passes.layout.vf2_utils import ErrorMap

from q_audit.passes import VF2_ERROR_MAP_KEY, error_map_is_supported


class _Inject(AnalysisPass):
    def __init__(self, error_map):
        super().__init__()
        self._map = error_map

    def __hash__(self):  # qiskit hashes every pass argument
        return id(self)

    def run(self, dag):
        self.property_set[VF2_ERROR_MAP_KEY] = self._map
        return dag


def _neighbourhood(coupling, centre: int, hops: int) -> set[int]:
    adjacency: dict[int, set[int]] = {}
    for a, b in coupling:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    seen = {centre}
    frontier = {centre}
    for _ in range(hops):
        nxt: set[int] = set()
        for node in frontier:
            nxt |= adjacency.get(node, set())
        seen |= nxt
        frontier = nxt
    return seen


def _skewed_map(snapshot, favoured: set[int], *, good=0.0001, bad=0.3):
    error_map = ErrorMap(snapshot.num_qubits + 2 * len(snapshot.edges))
    for q in range(snapshot.num_qubits):
        error_map.add_error((q, q), good if q in favoured else bad)
    for a, b in snapshot.coupling:
        value = good if (a in favoured and b in favoured) else bad
        error_map.add_error((a, b), value)
        error_map.add_error((b, a), value)
    return error_map


def _vf2_layout(circuit, target, error_map, seed=42):
    passes = [] if error_map is None else [_Inject(error_map)]
    passes.append(
        VF2Layout(
            target=target,
            seed=seed,
            call_limit=None,
            time_limit=None,
            max_trials=0,
            strict_direction=False,
        )
    )
    pm = PassManager(passes)
    pm.run(circuit)
    layout = pm.property_set.get("layout")
    if layout is None:
        return None
    return [layout[q] for q in circuit.qubits]


@pytest.fixture
def line5(snapshot):
    """5 qubit line. embeds into heavy-hex loads of different ways so whichever
    one vf2 picks is decided entirely by the error map"""
    from q_audit.feature_extract import unroll_to_basis

    qc = QuantumCircuit(5)
    qc.h(0)
    for i in range(4):
        qc.cx(i, i + 1)
    return unroll_to_basis(qc, snapshot.basis_gates)


# ---------------------------------------------------------------------------
# the contract itself
# ---------------------------------------------------------------------------


def test_errormap_injection_contract(line5, snapshot, target):
    """the main canary. ErrorMap still works and still steers VF2Layout"""
    # 1. the ErrorMap api hasn't changed
    error_map = ErrorMap(2)
    error_map.add_error((0, 0), 0.25)
    assert error_map.get((0, 0)) == pytest.approx(0.25)
    assert len(error_map) == 1

    # 2. both vf2 passes still read the property set key
    assert error_map_is_supported()

    # 3. injecting actually changes the answer, and changes it TO the region we
    #    favoured. "something different" isn't good enough, that could be luck
    island_a = _neighbourhood(snapshot.coupling, 100, 4)
    island_b = _neighbourhood(snapshot.coupling, 20, 4)
    assert not (island_a & island_b), "islands must be disjoint for this test to mean anything"

    layout_a = _vf2_layout(line5, target, _skewed_map(snapshot, island_a))
    layout_b = _vf2_layout(line5, target, _skewed_map(snapshot, island_b))

    assert layout_a is not None and layout_b is not None
    assert set(layout_a) <= island_a, f"injection ignored: {layout_a} not inside island A"
    assert set(layout_b) <= island_b, f"injection ignored: {layout_b} not inside island B"
    assert set(layout_a) != set(layout_b)


def test_uninjected_and_injected_layouts_differ(line5, snapshot, target):
    """with no injection vf2 uses qiskit's own map, ours has to be able to beat it"""
    default = _vf2_layout(line5, target, None)
    island = _neighbourhood(snapshot.coupling, 20, 4)
    injected = _vf2_layout(line5, target, _skewed_map(snapshot, island))
    assert default is not None and injected is not None
    assert set(default) != set(injected)


def test_property_set_key_name_is_stable():
    assert VF2_ERROR_MAP_KEY == "vf2_avg_error_map"


# ---------------------------------------------------------------------------
# VF2PostLayout and the strict_direction trap
# ---------------------------------------------------------------------------


def _post_layout(routed, target, error_map, *, strict: bool, seed=42):
    passes = [] if error_map is None else [_Inject(error_map)]
    passes.append(
        VF2PostLayout(
            target=target,
            seed=seed,
            strict_direction=strict,
            call_limit=None,
            time_limit=None,
            max_trials=0,
        )
    )
    pm = PassManager(passes)
    pm.run(routed)
    return pm.property_set.get("post_layout"), str(
        pm.property_set.get("VF2PostLayout_stop_reason")
    )


@pytest.fixture
def routed_line5(fake_backend):
    qc = QuantumCircuit(5)
    qc.h(0)
    for i in range(4):
        qc.cx(i, i + 1)
    pm = generate_preset_pass_manager(
        optimization_level=1,
        backend=fake_backend,
        seed_transpiler=7,
        initial_layout=[0, 1, 2, 3, 4],
    )
    return pm.run(qc)


def test_vf2_post_layout_non_strict_responds_to_injection(
    routed_line5, snapshot, target
):
    """strict_direction=False is the only branch that reads vf2_avg_error_map"""
    island = _neighbourhood(snapshot.coupling, 100, 4)
    assert not (island & {0, 1, 2, 3, 4})

    post, reason = _post_layout(
        routed_line5, target, _skewed_map(snapshot, island), strict=False
    )
    assert post is not None, f"VF2PostLayout found nothing ({reason})"
    moved = {
        p for q, p in post.get_virtual_bits().items()
        if routed_line5.find_bit(q).index in {0, 1, 2, 3, 4}
    }
    assert moved <= island, f"post-layout ignored the injected map: {sorted(moved)}"


def test_vf2_post_layout_strict_ignores_injection(routed_line5, snapshot, target):
    """the strict branch scores against the target directly and ignores us.

    this is a trap not a bug. if we ever let a strict VF2PostLayout run after
    picking a layout it would throw the whole injection away without saying
    anything. that's why the audited path pins initial_layout, which strips
    every VF2PostLayout out of the preset
    """
    island = _neighbourhood(snapshot.coupling, 100, 4)
    strict_injected, _ = _post_layout(
        routed_line5, target, _skewed_map(snapshot, island), strict=True
    )
    strict_plain, _ = _post_layout(routed_line5, target, None, strict=True)

    def as_indices(layout):
        if layout is None:
            return None
        return sorted(layout.get_virtual_bits().values())

    assert as_indices(strict_injected) == as_indices(strict_plain)


def test_preset_drops_vf2_post_layout_when_initial_layout_is_pinned(fake_backend):
    """guards the assumption the whole pinned strategy is built on"""

    def pass_names(pm):
        found: list[str] = []

        def walk(obj):
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    walk(item)
                return
            for attr in ("tasks", "_tasks", "passes"):
                if hasattr(obj, attr):
                    walk(list(getattr(obj, attr)))
                    return
            found.append(type(obj).__name__)

        for stage in pm.stages:
            sub = getattr(pm, stage)
            if sub is not None:
                walk(sub._tasks)
        return found

    free = generate_preset_pass_manager(
        optimization_level=3, backend=fake_backend, seed_transpiler=42
    )
    pinned = generate_preset_pass_manager(
        optimization_level=3,
        backend=fake_backend,
        seed_transpiler=42,
        initial_layout=[0, 1, 2, 3, 4],
    )
    assert "VF2PostLayout" in pass_names(free)
    assert "VF2PostLayout" not in pass_names(pinned)


# ---------------------------------------------------------------------------
# how vf2 weights the diagonal, which is where map_scaling=normalized comes from
# ---------------------------------------------------------------------------


def test_vf2_diagonal_exponent_is_one_qubit_op_count(snapshot, target):
    """the diagonal is a per 1q op rate, not a per qubit total.

    vf2 scores an embedding as prod_q (1-diag[q])**n1q(q). so put two couplers
    against each other. E1 has diag 0.010 each and edge 0.001, E2 has diag
    0.005 each and edge 0.020. E1 wins while there are few 1q ops and E2 wins
    once the diagonal has been exponentiated enough. crossover works out at
    n = 3.81 so E1 should win at n<=3 and E2 at n>=4.

    if qiskit ever switches to applying the diagonal once then E1 wins for
    every n and this fails, and at that point map_scaling="normalized" is wrong
    """
    edges = sorted({(min(a, b), max(a, b)) for a, b in snapshot.coupling})
    e1, e2 = edges[5], edges[20]
    assert not (set(e1) & set(e2))

    error_map = ErrorMap(snapshot.num_qubits + 2 * len(snapshot.coupling))
    for q in range(snapshot.num_qubits):
        error_map.add_error((q, q), 0.5)
    for a, b in snapshot.coupling:
        error_map.add_error((a, b), 0.5)
        error_map.add_error((b, a), 0.5)
    for q in e1:
        error_map.add_error((q, q), 0.010)
    for q in e2:
        error_map.add_error((q, q), 0.005)
    error_map.add_error(e1, 0.001)
    error_map.add_error((e1[1], e1[0]), 0.001)
    error_map.add_error(e2, 0.020)
    error_map.add_error((e2[1], e2[0]), 0.020)

    def winner(n_1q: int) -> str:
        qc = QuantumCircuit(2)
        for _ in range(n_1q):
            qc.sx(0)
        qc.ecr(0, 1)
        chosen = _vf2_layout(qc, target, error_map, seed=1)
        assert chosen is not None
        if set(chosen) == set(e1):
            return "E1"
        if set(chosen) == set(e2):
            return "E2"
        return f"other:{chosen}"

    assert [winner(n) for n in (1, 2, 3)] == ["E1", "E1", "E1"]
    assert [winner(n) for n in (4, 5, 10)] == ["E2", "E2", "E2"]


def test_barriers_do_not_count_toward_the_exponent(snapshot, target):
    """same mechanism, sanity check on dag_op_stats"""
    from q_audit.feature_extract import dag_op_stats

    qc = QuantumCircuit(2)
    qc.sx(0)
    qc.rz(0.5, 0)
    qc.barrier()
    qc.ecr(0, 1)
    stats = dag_op_stats(qc)
    assert stats.one_qubit_ops == 2  # sx + rz, barrier doesn't count
    assert stats.error_bearing_1q_ops == 1  # rz is free
    assert stats.active_qubits == 2
