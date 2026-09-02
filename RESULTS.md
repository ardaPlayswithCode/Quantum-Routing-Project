# results

ran the benchmark on fake_sherbrooke with 20 seeds and 16384 shots, alpha=1.0,
normalized map scaling. numbers below are hellinger fidelity vs an exact
statevector reference, and the counts in brackets are how many of the 20 paired
seeds the audited layout won.

ghz7: +2.43% fidelity (20/20 seeds won)
ghz10: +3.33% (20/20)
qft7: no real change (expected, uniform output distribution so fidelity can't see anything)
qft_mirror7: +1.80% (12/20)
su2_6x3: +0.79% (20/20)
qaoa6: +1.92% (20/20)
random7d20: +1.82% (20/20)

six of the seven are significant at 95%. qft7 is the one that isn't, which is
exactly what its sharp=False flag says should happen.

## the control arm

the control is the same relocation machinery but using qiskit's own error map
instead of ours, no T1/T2 at all. i added this because otherwise you can't tell
whether the gain comes from the physics or just from running a non-strict
VF2PostLayout, which the preset never does on its own.

the control actually made things worse on 5 of the 7 circuits, badly so on the
dense ones (-54% esp on qft_mirror7, -32% on qft7). the audited arm beat the
control on all seven and every one was significant. so the T1/T2 info is what
matters, it isn't just "run VF2PostLayout twice".

## alpha

tried 0.0, 0.5, 1.0, 2.0 and 4.0. mean paired fidelity delta over ghz7 /
su2_6x3 / qaoa6 / random7d20:

0.0 -> +0.0176
0.5 -> +0.0176
1.0 -> +0.0232
2.0 -> +0.0149
4.0 -> +0.0110

so 1.0 works best and that's the default. past about 2 it starts trading gate
and readout quality away for coherence and ghz7 ends up worse than baseline.

## raw vs normalized

normalized is worth roughly double: mean paired fidelity delta +0.0244 vs
+0.0123 for raw. raw also turns ghz7 into a small loss (-0.89% esp) where
normalized gains +0.97%.

## the one number i'd show someone

worst T2 on the baseline layout for qft7 was 46us, on the audited layout it was
218us. that's the whole point really, qiskit put the circuit on a qubit that
had dephased before the circuit finished and it had no way of knowing.

## reproducing

uv pip install -e ".[dev]"
pytest
python benchmarks/bench.py --seeds 20 --shots 16384

add --control for the control arm, or --alphas 0.0 0.5 1.0 2.0 4.0 for the
alpha sweep. takes a while, the aer simulations are the slow part.

one caveat: qiskit 2.5.1's preset pass manager isn't deterministic at
optimization level 2 or 3 even with a fixed seed, so individual per-seed
numbers move a bit between runs. best-of-20 is stable, don't quote a single
seed.
