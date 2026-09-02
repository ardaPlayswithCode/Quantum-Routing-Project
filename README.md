# q-audit

transpiles a circuit twice against the same backend and shows you both side by
side: once with stock qiskit (`optimization_level=3`) and once with a T1/T2
aware error map injected into vf2's layout scoring.

the point is that qiskit's layout scoring uses gate error and readout error but
doesn't know about coherence times. a qubit with a 2.6us T2 has dephased long
before a 30us circuit finishes, and nothing in the default pipeline notices.
this measures what that costs you.

## install

```
uv venv
uv pip install -e '.[dev]'
```

then either `q-audit run ...` or `./q-audit run ...` (see the macos note at the
bottom if the installed one doesn't work).

## running it

```
./q-audit run examples/ghz7.py
./q-audit run examples/qft7.py --backend fake_sherbrooke --seeds 20
./q-audit run examples/ghz7.py --json | jq .comparison
./q-audit calibration fake_sherbrooke
```

circuits can be a .py file that defines `qc`, a .qasm file, or `-` for qasm on
stdin. `--json` gives you one json document on stdout and nothing else, even
when it fails, so you can pipe it. progress goes to stderr as ndjson. exit
codes are 0 ok, 2 user error, 3 calibration problem, 4 transpiler problem.

useful flags: `--alpha` (weight on the idle penalty, default 1.0), `--seeds`,
`-O`, `--map-scaling raw|normalized`, `--no-post-layout`, `--verify` (also
simulates both with aer, slow), `--allow-stale`.

## how it works

sabre's swap heuristic is in rust and has no python hook, so you can't make
routing noise aware from python. but VF2Layout and VF2PostLayout both read an
optional error map out of the property set under `vf2_avg_error_map`, so that's
where the T1/T2 goes. we pick better qubits and let stock sabre route between
them.

for each idle gap on each wire we charge

```
eps_idle(t) = (3 - 2*exp(-t/T2) - exp(-t/T1)) / 6
```

which is the standard thermal relaxation infidelity. it matches aer's
`thermal_relaxation_error` to machine precision, there's a test for that. only
idle gaps get charged, gate errors from calibration already include whatever
decohered during the gate. estimated success probability is then just
`prod(1 - eps)` over everything.

there are two strategies and it tries both and keeps whichever scores higher.
`vf2_pinned` is for sparse circuits where vf2 finds an exact embedding, it pins
that as the initial layout and gets zero routing overhead.
`post_layout_relocate` is for dense circuits, it takes the baseline's routed
circuit and moves it onto better physical qubits without touching the gates or
the swaps.

both paths use the same input circuit, the same pass manager, the same seed
list and the same objective, otherwise the comparison would be meaningless.

## results

short version: +0.8% to +3.3% hellinger fidelity depending on the circuit, on
20 paired seeds against an exact statevector reference. see RESULTS.md for the
numbers and the control arm, which is the part i'd actually look at.

## what doesn't work / caveats

- only tested on `fake_sherbrooke`. the live backend path is written but i have
  never run it against real hardware, so treat that as untested.
- the `vf2_avg_error_map` property set key isn't public qiskit api. if an
  upgrade changes it the audit doesn't crash, it just silently stops doing
  anything, which is worse. tests/test_injection_contract.py is there to catch
  that, and it's why qiskit is pinned exactly.
- selecting the best circuit by esp makes the esp improvement partly self
  fulfilling, since esp is also what the injected map optimises. that's why the
  benchmark measures hellinger fidelity with aer instead. esp is the model,
  the simulation is the measurement.
- qiskit 2.5.1's preset pass manager is not deterministic at optimization level
  2 or 3, even with a fixed `seed_transpiler`. run it twice on the same circuit
  in the same process and you can get different layouts and different depths.
  opt level 1 is fine. it isn't thread count and it isn't hash randomisation, i
  checked both. so best-of-N is reproducible but a single seed's numbers aren't.
- circuits with classical control flow are rejected unless you pass
  `--allow-control-flow`, because the idle model assumes one static schedule.
- no 3+ qubit gates after unrolling. vf2 gives up on those without telling you.

## tests and benchmarks

```
pytest
pytest -m "not slow"
python benchmarks/bench.py --seeds 20 --shots 16384
python benchmarks/bench.py --seeds 20 --control
python benchmarks/bench.py --write-golden
```

162 tests. the interesting ones are tests/test_injection_contract.py (does the
qiskit hook still behave) and tests/test_physics_vs_aer.py (is the formula
right).

## macos thing that cost me an afternoon

this repo lives under `~/Desktop`, and macos sets the `UF_HIDDEN` flag on
everything inside `.venv` there. python's `site.addpackage()` silently skips
hidden `.pth` files, so `uv pip install -e .` appears to work but the editable
path never reaches `sys.path` and `import q_audit` fails with no explanation at
all.

check with `ls -lO .venv/lib/python3.12/site-packages/*.pth` and look for
`hidden`. fix it with

```
chflags nohidden .venv/lib/python3.12/site-packages/*.pth
```

or just use the `./q-audit` launcher, which sets PYTHONPATH and sidesteps the
whole thing.
