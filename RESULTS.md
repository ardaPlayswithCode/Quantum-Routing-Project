# results

ran the benchmark on fake_sherbrooke with 20 seeds and 16384 shots, alpha=1.0,
normalized map scaling. fidelity is hellinger fidelity of the aer output
against the ideal output distribution. everything here is simulation. nothing
has been run on real hardware.

two ways of reading the same runs, and they are not the same number:

- best-of-20 (dF, dESP): for each arm, take the best of the 20 seeds, then
  compare. this is what the cli reports.
- paired dF: hold the routing seed fixed, compare the two layouts on each seed,
  average the 20 differences. this isolates the layout effect and has a 95% CI.
  "wins" is how many of the 20 paired seeds the audited layout won.

| circuit      | dESP (best-of-20) | dF (best-of-20) | paired dF [95% CI]        | wins  | strategy      |
|--------------|-------------------|-----------------|---------------------------|-------|---------------|
| ghz7         | +0.97%            | +2.43%          | +0.0223 [+0.0214, +0.0232] | 20/20 | vf2_pinned    |
| ghz10        | +0.51%            | +3.33%          | +0.0286 [+0.0275, +0.0297] | 20/20 | relocate      |
| qft7         | +0.00%            | +0.00%          | +0.0004 [-0.0000, +0.0008] | 11/20 | relocate      |
| qft_mirror7  | +1.31%            | +1.80%          | +0.0290 [+0.0128, +0.0452] | 12/20 | relocate      |
| su2_6x3      | +3.95%            | +0.79%          | +0.0074 [+0.0060, +0.0089] | 20/20 | relocate      |
| qaoa6        | +13.55%           | +1.92%          | +0.0197 [+0.0192, +0.0202] | 20/20 | relocate      |
| random7d20   | +5.47%            | +1.82%          | +0.0425 [+0.0332, +0.0517] | 20/20 | relocate      |

six of the seven are significant at 95%. qft7 is the one that isn't. its ideal
output is uniform and noise also pushes it toward uniform, so hellinger
fidelity sits near 0.997 however badly it's compiled. the metric can't see
anything there, so read it as "no information", not "no effect".

the gains are small (+0.8% to +3.3% best-of-20 fidelity) and come from one
simulated chip. esp gains are larger than fidelity gains, but esp is also what
the injected map optimises, so treat fidelity as the measurement and esp as the
model.

## the control arm

the preset pass manager only ever runs a strict VF2PostLayout. ours runs a
non-strict one, which searches a much bigger space, so some of the gain could be
that alone. the control arm runs the identical relocation machinery with
qiskit's own error map and no injection.

| circuit      | control vs baseline (esp) | audited vs control (esp) | dF (audited - control) |
|--------------|---------------------------|--------------------------|------------------------|
| ghz7         | -2.99%                    | +4.09%                   | +0.0370                |
| ghz10        | -1.08%                    | +1.61%                   | +0.0428                |
| qft7         | -31.96%                   | +46.98%                  | +0.0014                |
| qft_mirror7  | -54.05%                   | +120.46%                 | +0.1574                |
| su2_6x3      | +1.07%                    | +2.86%                   | +0.0050                |
| qaoa6        | +6.82%                    | +6.30%                   | +0.0138                |
| random7d20   | -23.51%                   | +37.88%                  | +0.0642                |

every dF here is significant at 95%. a non-strict relocation driven by qiskit's
own map is worse than baseline on five of the seven circuits, badly so on the
dense ones. so the gain is not just "run VF2PostLayout non-strictly", and the
injected map is what turns it into a win.

what the control does not tell you is how much of that comes from the T1/T2
term specifically. the injected map also carries gate and readout error, scaled
differently from qiskit's. see the alpha sweep below, where turning the idle
term off still keeps most of the gain.

## alpha

tried 0.0, 0.5, 1.0, 2.0 and 4.0 on ghz7, su2_6x3, qaoa6 and random7d20. mean
paired dF over those four:

| alpha | mean paired dF | mean paired dESP | significant |
|-------|----------------|------------------|-------------|
| 0.0   | +0.0176        | +0.058           | 4/4         |
| 0.5   | +0.0176        | +0.066           | 3/4         |
| 1.0   | +0.0232        | +0.067           | 4/4         |
| 2.0   | +0.0149        | +0.055           | 4/4         |
| 4.0   | +0.0110        | +0.051           | 4/4         |

1.0 works best, so that's the default. over-weighting the idle term trades gate
and readout quality for coherence and makes ghz7 worse than baseline.

the thing to notice is the alpha=0 row. with the idle term off, mean paired dF
is still +0.0176, about three quarters of the +0.0232 at alpha=1.0. so most of
the gain survives without any T1/T2 information, and the idle term adds roughly
+0.006 on top. the rest comes from the injected map and relocation machinery
(including how the map is scaled, below). this is four circuits and one chip,
so i wouldn't read much finer than that.

## raw vs normalized

normalized is worth roughly double raw: mean paired dF +0.0244 vs +0.0123, over
ghz7, su2_6x3, qaoa6, random7d20 and qft_mirror7. the reason is that vf2
exponentiates the diagonal of the error map by the circuit's 1-qubit op count,
so the raw formula multiplies the one-shot readout and idle penalties by an
operation rate and swamps the gate error. normalized corrects for that.

## worked example: examples/qft7.py

this is the example file (a hadamard layer followed by qft-7, so the ideal
output is a delta at |0...0>), not the uniform-output qft7 in the benchmark
table above.

| metric                    | baseline  | audited   | change |
|---------------------------|-----------|-----------|--------|
| esp                       | 0.4695    | 0.4955    | +5.54% |
| hellinger fidelity (aer)  | 0.5163    | 0.5223    | +1.16% |
| worst T2 on layout        | 46.2 us   | 218.2 us  | +372%  |
| routing swaps             | 15        | 12        | -20%   |
| two-qubit gates           | 67        | 66        | -1.5%  |
| depth                     | 254       | 233       | -8.3%  |
| duration                  | 31.42 us  | 30.55 us  | -2.8%  |

strategy was post_layout_relocate. the audited layout avoids the 46 us qubit,
and the fidelity gain is about 1.2%. i'd show this one as an illustration of
what the tool does, not as evidence of how big the effect is.

## reproducing

```
uv pip install -e ".[dev]"
pytest
python benchmarks/bench.py --seeds 20 --shots 16384
```

add `--control` for the control arm, or
`--alphas 0.0 0.5 1.0 2.0 4.0 --circuits ghz7 su2_6x3 qaoa6 random7d20` for the
alpha sweep, or `--scaling normalized raw` for the scaling comparison. takes a
while, the aer simulations are the slow part.

one caveat: qiskit 2.5.1's preset pass manager isn't deterministic at
optimization level 2 or 3 even with a fixed seed, so individual per-seed numbers
move between runs. best-of-20 is stable (identical to nine decimals across four
concurrent 20-seed runs), so don't quote a single seed.
