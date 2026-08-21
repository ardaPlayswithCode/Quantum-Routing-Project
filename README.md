# q-audit — quantum transpile audit

`q-audit` transpiles a circuit **twice** against the same backend and shows you the
difference side by side:

* **baseline** — stock Qiskit, `generate_preset_pass_manager(optimization_level=3)`
* **audited** — the same pipeline, but with a **T1/T2-aware error map injected into
  VF2's layout scoring**, so the circuit lands on qubits that will still be coherent
  when the last gate lands

Qiskit's default layout scoring uses gate and readout error. It does **not** know that
a qubit with a 2.6 µs T2 will have dephased long before a 30 µs circuit finishes.
That is the gap this tool measures.

```
q-audit run examples/qft7.py --backend fake_sherbrooke
```

## Why the layout, and not SABRE

SABRE's swap-selection heuristic lives in Rust and exposes no Python hook — there is
no supported way to make routing noise-aware from Python. `VF2Layout` and
`VF2PostLayout` *do* read an optional `ErrorMap` out of the pass manager's property
set (`vf2_avg_error_map`). So the leverage point is **layout**, not routing: bias
where the circuit lands, then let stock SABRE route between those qubits.

That hook is a private-ish contract. `tests/test_injection_contract.py` is a canary
suite that fails loudly if a Qiskit upgrade changes it — because the failure mode is
not a crash, it is a silent no-op audit.

## The physics

A qubit idling for time `t` accumulates average infidelity

```
eps_idle(t) = (3 - 2*exp(-t/T2) - exp(-t/T1)) / 6
```

(limits: `eps(0) = 0`, `eps(inf) = 1/2`). This matches Aer's
`thermal_relaxation_error` to machine precision — see `tests/test_physics_vs_aer.py`.

Decoherence is charged to **idle windows only** — the gaps between consecutive
instructions on a wire, which naturally includes the tail before a measurement.
A reported gate error already contains the decoherence that happened *during* the
gate, so adding a relaxation term on top would double count. Time before a qubit's
first instruction is not charged either: a qubit sitting in |0⟩ has nothing to damp
and no superposition to dephase.

Estimated Success Probability is then `prod(1 - eps_gate) * prod(1 - eps_idle)`.

## The two audited strategies

`q-audit` tries both and reports whichever scores higher, along with which one won:

| strategy | when it applies | what it does |
| --- | --- | --- |
| `vf2_pinned` | VF2Layout finds an exact subgraph embedding (sparse circuits) | pins that layout as `initial_layout`; zero routing overhead |
| `post_layout_relocate` | always (unless `--no-post-layout`) | takes the baseline's *routed* circuit and relocates it onto better qubits with `VF2PostLayout(strict_direction=False)`; gate structure, SWAPs and depth are untouched |

`strict_direction=False` matters: the `True` branch scores against the target's exact
per-edge errors and ignores the injected map entirely. It also means a relocation can
land a 2q gate on the uncalibrated orientation of a coupler, so the relocate path
re-runs the same `GateDirection` fix-up the preset translation stage uses.

Pinning `initial_layout` has a second, load-bearing effect: it removes **every**
`VF2PostLayout` from the preset pass manager, including the `strict_direction=True`
one in the optimization stage that would otherwise silently overwrite the audited
layout using Qiskit's own error map.

## Fairness

The comparison is only worth reading if both sides get the same budget:

* both paths transpile the **same bound circuit object**
* both run the **same preset pass manager** at the same optimization level
* both sweep the **same seed list** (`--seeds N`, starting at `--base-seed`)
* both select their winner by the **same objective** (ESP)

The report also shows the stock single-seed baseline, so you can see how much of the
gap is the audit and how much is just sampling 20 seeds.

**Stated caveat:** selecting by ESP makes the headline ESP delta partly
self-fulfilling — ESP is also what the injected map optimises. The independent check
is `benchmarks/bench.py`, which simulates both circuits with Aer and compares
Hellinger fidelity against an exact statevector reference. ESP is the model;
Hellinger is the measurement.

## Install

```bash
uv venv
uv pip install -e '.[dev]'
```

Runtime deps are pinned exactly (`qiskit==2.5.1`, `qiskit-ibm-runtime==0.48.0`).
`qiskit-aer==0.17.2` is a **dev** dependency: it is imported lazily inside `--verify`
and the benchmark harness only, never on the optimizer path.

### Known environment gotcha (macOS)

CPython's `site.addpackage()` **silently skips `.pth` files carrying the macOS
`UF_HIDDEN` flag**. Under `~/Desktop` (and other synced locations) that flag gets
applied to everything inside `.venv`, which makes `uv pip install -e .` inert: the
editable path never reaches `sys.path` and `import q_audit` fails with no diagnostic.

Check with `ls -lO .venv/lib/python3.12/site-packages/*.pth` (look for `hidden`).
Either clear it:

```bash
chflags nohidden .venv/lib/python3.12/site-packages/*.pth
```

or just use the bundled launcher, which sidesteps `.pth` entirely:

```bash
./q-audit run examples/qft7.py
```

## Usage

```bash
q-audit run CIRCUIT [options]

  CIRCUIT                  .py defining `qc`, a .qasm file, or '-' for QASM on stdin
  --backend, -b            backend name (default: fake_sherbrooke)
  --json                   emit one JSON document on stdout instead of a table
  --alpha FLOAT            weight on the idle-decoherence penalty (default: 1.0)
  --seeds INT              transpiler seeds per path (default: 20)
  --base-seed INT          first seed (default: 42)
  -O, --optimization-level 0..3 (default: 3)
  --schedule asap|alap     scheduling used for idle analysis (default: asap)
  --map-scaling            normalized | raw (default: normalized)
  --no-post-layout         skip the relocate strategy
  --no-topology            skip the ASCII device map
  --allow-stale            proceed on calibration older than 24h
  --allow-control-flow     audit circuits containing if/for/switch
  --verify                 also simulate both circuits with Aer (needs qiskit-aer)
  --refresh                ignore the calibration cache
  -q, --quiet              silence stderr progress

q-audit calibration [BACKEND] [--json]    inspect the snapshot that would be used
q-audit version                           versions + injection-hook health check
```

### `--map-scaling`, and why `normalized` is the default

VF2 scores a candidate embedding as

```
1 - prod_q (1 - diag[q]) ** n1q(q) * prod_e (1 - edge[e]) ** n2q(e)
```

where `n1q(q)` counts **every** 1-qubit DAG op on that wire — zero-duration `rz`
frame changes and `measure` included, barriers excluded. The diagonal is therefore a
*per-1q-operation* rate, not a per-qubit total (verified empirically in
`test_vf2_diagonal_exponent_is_one_qubit_op_count`).

`raw` uses the naive `sx + readout + alpha*idle`, which multiplies the one-shot
readout and idle penalties by however many 1q ops happen to land on that wire.
`normalized` spreads them across the exponent instead:

```
diag[q] = 1 - exp( ( g_bar*ln(1-sx) + m_bar*ln(1-ro) + ln(1-alpha*eps_idle) ) / n_bar )
```

Measured over ghz7 / su2_6x3 / qaoa6 / random7d20 / qft_mirror7 at 20 seeds,
`normalized` is worth roughly double: mean paired fidelity delta **+0.0244**
versus **+0.0123** for `raw`, and `raw` turns GHZ-7 into a small *loss*
(-0.89% ESP) where `normalized` gains +0.97%.

## Reproducibility, and its ceiling

Everything q-audit controls is deterministic: every transpile passes an explicit
`seed_transpiler`, both paths use the identical seed list, parameters are bound
from a hash of (circuit name, parameter name), and the VF2 search is bounded by
`call_limit`/`max_trials` — never by a wall-clock `time_limit`, which would make
the result depend on machine load.

What q-audit does **not** control is upstream. On Qiskit 2.5.1,
`generate_preset_pass_manager(optimization_level=3, seed_transpiler=42)` run
repeatedly on the same circuit *in the same process* returns more than one
distinct result — different layouts and different depths:

```python
pm = generate_preset_pass_manager(optimization_level=3, backend=FakeSherbrooke(),
                                  seed_transpiler=42)
{(tuple(pm.run(qc).layout.initial_index_layout(filter_ancillas=True)),
  pm.run(qc).depth()) for _ in range(8)}
# -> {((125,124,123,122,121,120), 58), ((120,121,122,123,124,125), 56)}
```

`optimization_level=1` is stable; 2 and 3 are not. `VF2Layout`, `SabreLayout`
and `VF2PostLayout` are each deterministic in isolation, so it is something in
the O2/O3 optimization loop. It is not thread count (`RAYON_NUM_THREADS=1` does
not fix it) and not hash randomisation (`PYTHONHASHSEED=0` does not fix it).

Practical consequence: **the best-of-N headline is stable across runs; an
individual seed's numbers may not be.** Across four concurrent 20-seed runs of
`su2_6x3` the reported best ESP was identical to nine decimal places every time,
while one of the twenty per-seed values moved. Report best-of-N; do not quote a
single seed as if it were reproducible.

## Output contract

* **stdout** — the report; with `--json`, exactly one JSON document, *including on
  failure* (`{"ok": false, "error": {...}}`), so `q-audit ... --json | jq` never sees
  a truncated stream
* **stderr** — NDJSON progress events, one object per line
* **exit codes** — `0` ok, `2` user error, `3` calibration error, `4` transpiler error

## Calibration

Resolution order: `--backend fake_*` (offline reference snapshot) → on-disk JSON cache
(1 hour TTL, under `platformdirs.user_cache_dir`) → live `QiskitRuntimeService`.

We never cache Qiskit objects — only a `CalibrationSnapshot` of plain numbers, which
survives a Qiskit upgrade. `T2` is clamped to `2*T1` at ingest. Snapshot age is
printed in every report; live/cached data older than 24h is **refused** unless you
pass `--allow-stale`. Offline reference snapshots (fake backends) are never refused —
being frozen is the point of them — but they always carry a loud warning.

## What the benchmarks actually show

Measured on `fake_sherbrooke`, 20 paired transpiler seeds, 16384 Aer shots,
`alpha=1.0`, `--map-scaling normalized`. `dF` is the Hellinger fidelity delta
against an exact statevector reference; CIs are 95% over the paired seeds.

| circuit | best-of-20 dESP | best-of-20 dF | paired dF (95% CI) | wins | strategy |
| --- | --- | --- | --- | --- | --- |
| ghz7 | +0.97% | +2.43% | +0.0223 [+0.0214, +0.0232] | 20/20 | `vf2_pinned` |
| ghz10 | +0.51% | +3.33% | +0.0286 [+0.0275, +0.0297] | 20/20 | `post_layout_relocate` |
| qft7 | +0.00% | +0.00% | +0.0004 [-0.0000, +0.0008] | 11/20 | `post_layout_relocate` |
| qft_mirror7 | +1.31% | +1.80% | +0.0290 [+0.0128, +0.0452] | 12/20 | `post_layout_relocate` |
| su2_6x3 | +3.95% | +0.79% | +0.0074 [+0.0060, +0.0089] | 20/20 | `post_layout_relocate` |
| qaoa6 | +13.55% | +1.92% | +0.0197 [+0.0192, +0.0202] | 20/20 | `post_layout_relocate` |
| random7d20 | +5.47% | +1.82% | +0.0425 [+0.0332, +0.0517] | 20/20 | `post_layout_relocate` |

Six of seven are significant; `qft7` is the one that is not, exactly as its
`sharp=False` flag predicts (a near-uniform reference cannot resolve a fidelity
change).

### Choosing alpha

Sweeping `--alphas 0.0 0.5 1.0 2.0 4.0` over ghz7 / su2_6x3 / qaoa6 /
random7d20, mean paired fidelity delta:

| alpha | mean dF | mean dESP | significant |
| --- | --- | --- | --- |
| 0.0 | +0.0176 | +0.058 | 4/4 |
| 0.5 | +0.0176 | +0.066 | 3/4 |
| **1.0** | **+0.0232** | **+0.067** | 4/4 |
| 2.0 | +0.0149 | +0.055 | 4/4 |
| 4.0 | +0.0110 | +0.051 | 4/4 |

`alpha=1.0` is the empirical optimum and is the default. Over-weighting the idle
term (`alpha>=2`) starts trading away gate and readout quality for coherence and
makes GHZ-7 *worse* than baseline.

### How much of the gain is actually the T1/T2 injection?

`--control` adds a third arm that runs the **same relocation machinery with
Qiskit's own error map** and no injection. This matters, because the preset pass
manager only ever runs a *strict* `VF2PostLayout` in its optimization stage,
which searches a much smaller space — so some of any gain could be "we ran a
non-strict post-layout at all", not "we know about T1/T2".

```
python benchmarks/bench.py --seeds 20 --control
```

20 seeds, 16384 shots. `dF` is the Aer fidelity delta of audited over control:

| circuit | control vs baseline | audited vs control | dF (audited − control) |
| --- | --- | --- | --- |
| ghz7 | -2.99% | +4.09% | +0.0370 SIG |
| ghz10 | -1.08% | +1.61% | +0.0428 SIG |
| qft7 | -31.96% | +46.98% | +0.0014 SIG |
| qft_mirror7 | -54.05% | +120.46% | +0.1574 SIG |
| su2_6x3 | +1.07% | +2.86% | +0.0050 SIG |
| qaoa6 | +6.82% | +6.30% | +0.0138 SIG |
| random7d20 | -23.51% | +37.88% | +0.0642 SIG |

The answer is unambiguous, and it is not the answer you might expect: a
non-strict relocation driven by **Qiskit's own error map is actively harmful**
on five of seven circuits, catastrophically so on the dense ones (-54% on
qft_mirror7). It is the T1/T2 information that turns the relocation into a win —
the audited arm beats the control on all seven circuits, every one statistically
significant in simulated fidelity.

So the mechanism is doing real work; it is not a repackaged
"run VF2PostLayout twice" trick.

## Tests & benchmarks

```bash
pytest                      # full suite
pytest -m "not slow"        # skip end-to-end transpiles
pytest tests/test_injection_contract.py   # the canary suite

python benchmarks/bench.py --seeds 20 --shots 16384
python benchmarks/bench.py --alphas 0.0 0.5 1.0 2.0 4.0 --circuits ghz7 qaoa6
python benchmarks/bench.py --seeds 20 --control     # isolate what injection buys
python benchmarks/bench.py --write-golden           # refresh the regression gate
```

A note on the benchmark suite: QFT applied to |0…0⟩ produces a uniform
computational-basis distribution, and noise also produces something near-uniform, so
its Hellinger fidelity barely moves however badly it is compiled. Those cases are
marked `sharp=False` and kept for their compiler metrics only. The **mirror** circuits
(`U`, barrier, `U†`) have an exact delta-function reference and are the sensitive
fidelity probes.
