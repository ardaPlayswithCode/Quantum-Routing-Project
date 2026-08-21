#!/usr/bin/env python
"""Benchmark harness: does the injected layout actually help, and by how much?

Two comparisons are reported, because they answer different questions.

* **Paired, per seed** -- for each transpiler seed, transpile the circuit both
  ways and simulate both.  The routing seed is held fixed, so the paired delta
  isolates the effect of the layout choice.  Reported with a 95% CI over the
  seed list.  This is the honest statistical claim.
* **Best of N** -- what a user actually gets from ``q-audit``: the highest-ESP
  circuit each path produced.  Both paths get the same N and the same seeds.

ESP is our model; Hellinger fidelity against an exact statevector reference is
the independent measurement.  Where they disagree, believe the simulation.

Usage:
    python benchmarks/bench.py --seeds 20 --shots 16384
    python benchmarks/bench.py --alphas 0.1 0.5 1.0 2.0 --circuits ghz7 qft_mirror7
    python benchmarks/bench.py --write-golden
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from benchmarks.circuits import DEFAULT_SUITE, SUITE  # noqa: E402
from q_audit.calibration import load_fake_backend  # noqa: E402
from q_audit.feature_extract import extract_features  # noqa: E402
from q_audit.physics import paired_mean_ci  # noqa: E402
from q_audit.transpile_audit import (  # noqa: E402
    AuditSettings,
    run_audit,
    run_baseline_only,
    run_control_relocation,
)

GOLDEN_PATH = ROOT / "benchmarks" / "golden.json"


@dataclass
class CaseResult:
    name: str
    sharp: bool
    note: str
    num_qubits: int
    layout_source: str
    t_est_s: float
    vf2_stop_reason: str
    baseline_layout: list[int] = field(default_factory=list)
    audited_layout: list[int] = field(default_factory=list)
    esp_baseline: list[float] = field(default_factory=list)
    esp_audited: list[float] = field(default_factory=list)
    fid_baseline: list[float] = field(default_factory=list)
    fid_audited: list[float] = field(default_factory=list)
    esp_control: list[float] = field(default_factory=list)
    fid_control: list[float] = field(default_factory=list)
    control: dict = field(default_factory=dict)
    best: dict = field(default_factory=dict)
    paired: dict = field(default_factory=dict)
    seconds: float = 0.0


def _pct(new: float, old: float) -> float | None:
    return None if old == 0 else 100.0 * (new / old - 1.0)


def run_case(
    name: str,
    backend,
    snapshot,
    *,
    seeds: int,
    shots: int,
    alpha: float,
    scaling: str,
    simulate: bool,
    control: bool = False,
) -> CaseResult:
    started = time.monotonic()
    spec = SUITE[name]
    circuit = spec.build()
    bound, unrolled, features = extract_features(circuit, snapshot)
    settings = AuditSettings(
        alpha=alpha, seeds=seeds, map_scaling=scaling, keep_circuits=simulate
    )
    result = run_audit(bound, unrolled, features, snapshot, backend, settings)

    case = CaseResult(
        name=name,
        sharp=spec.sharp,
        note=spec.note,
        num_qubits=circuit.num_qubits,
        layout_source=result.layout_info["source"],
        t_est_s=features.t_est_s,
        vf2_stop_reason=result.layout_info["vf2_stop_reason"],
        baseline_layout=result.baseline.best.layout,
        audited_layout=result.audited.best.layout,
        esp_baseline=[m.esp for m in result.baseline.all_metrics],
        esp_audited=[m.esp for m in result.audited.all_metrics],
    )

    control_path = None
    if control:
        baseline_only, routed_circuits, routed_swaps = run_baseline_only(
            bound, backend, snapshot, settings
        )
        control_path = run_control_relocation(
            routed_circuits, routed_swaps, backend.target, snapshot, settings
        )
        case.esp_control = [m.esp for m in control_path.all_metrics]

    if simulate:
        from qiskit_aer.noise import NoiseModel

        from q_audit.verify import ensure_measured, ideal_distribution, simulated_fidelity

        ideal = ideal_distribution(ensure_measured(bound))
        noise_model = NoiseModel.from_backend(backend)
        arms = [
            (result.baseline, case.fid_baseline),
            (result.audited, case.fid_audited),
        ]
        if control_path is not None:
            arms.append((control_path, case.fid_control))
        for path, sink in arms:
            for seed, transpiled in zip(settings.seed_list(), path.all_circuits):
                fidelity, _ = simulated_fidelity(
                    transpiled,
                    ideal,
                    backend,
                    shots=shots,
                    seed_simulator=1000 + seed,
                    schedule_method=settings.schedule_method,
                    noise_model=noise_model,
                )
                sink.append(fidelity)

    base_best, audit_best = result.baseline.best, result.audited.best
    case.best = {
        "esp_baseline": base_best.esp,
        "esp_audited": audit_best.esp,
        "esp_gain_pct": _pct(audit_best.esp, base_best.esp),
        "depth_baseline": base_best.depth,
        "depth_audited": audit_best.depth,
        "two_qubit_baseline": base_best.two_qubit_gates,
        "two_qubit_audited": audit_best.two_qubit_gates,
        "swaps_baseline": base_best.swap_gates,
        "swaps_audited": audit_best.swap_gates,
        "duration_baseline_s": base_best.duration_s,
        "duration_audited_s": audit_best.duration_s,
        "idle_baseline_s": base_best.total_idle_s,
        "idle_audited_s": audit_best.total_idle_s,
        "worst_t2_baseline_s": base_best.worst_t2_on_layout_s,
        "worst_t2_audited_s": audit_best.worst_t2_on_layout_s,
        "esp_single_seed_baseline": result.baseline_default.esp,
    }
    if case.fid_baseline:
        best_index = case.esp_audited.index(max(case.esp_audited))
        base_index = case.esp_baseline.index(max(case.esp_baseline))
        case.best["fidelity_baseline"] = case.fid_baseline[base_index]
        case.best["fidelity_audited"] = case.fid_audited[best_index]
        case.best["fidelity_gain_pct"] = _pct(
            case.fid_audited[best_index], case.fid_baseline[base_index]
        )

    if control_path is not None:
        case.control = {
            "esp_best": control_path.best.esp,
            "esp_gain_pct_vs_baseline": _pct(control_path.best.esp, base_best.esp),
            "esp_gain_pct_audited_vs_control": _pct(
                audit_best.esp, control_path.best.esp
            ),
        }
        control_deltas = [a - b for a, b in zip(case.esp_audited, case.esp_control)]
        cmean, clo, chi = paired_mean_ci(control_deltas)
        case.control["esp_mean_delta_vs_control"] = cmean
        case.control["esp_ci95_vs_control"] = [clo, chi]
        if case.fid_control:
            fdeltas = [a - b for a, b in zip(case.fid_audited, case.fid_control)]
            fmean, flo, fhi = paired_mean_ci(fdeltas)
            case.control["fidelity_mean_delta_vs_control"] = fmean
            case.control["fidelity_ci95_vs_control"] = [flo, fhi]
            case.control["fidelity_significant_vs_control"] = flo > 0 or fhi < 0
            base_index = case.esp_control.index(max(case.esp_control))
            case.control["fidelity_best"] = case.fid_control[base_index]

    esp_deltas = [a - b for a, b in zip(case.esp_audited, case.esp_baseline)]
    mean, lo, hi = paired_mean_ci(esp_deltas)
    case.paired = {
        "n": len(esp_deltas),
        "esp_mean_delta": mean,
        "esp_ci95": [lo, hi],
        "esp_wins": sum(1 for d in esp_deltas if d > 0),
    }
    if case.fid_baseline:
        fid_deltas = [a - b for a, b in zip(case.fid_audited, case.fid_baseline)]
        fmean, flo, fhi = paired_mean_ci(fid_deltas)
        case.paired.update(
            {
                "fidelity_mean_delta": fmean,
                "fidelity_ci95": [flo, fhi],
                "fidelity_wins": sum(1 for d in fid_deltas if d > 0),
                "fidelity_significant": flo > 0 or fhi < 0,
            }
        )
    case.seconds = time.monotonic() - started
    return case


def print_summary(cases: list[CaseResult], *, alpha: float, scaling: str) -> None:
    print()
    print(f"=== alpha={alpha}  map_scaling={scaling} ===")
    header = (
        f"{'circuit':<18}{'layout src':<14}{'ESP base':>9}{'ESP aud':>9}{'dESP%':>8}"
        f"{'F base':>9}{'F aud':>9}{'dF%':>8}{'swaps':>8}{'depth':>12}{'sharp':>7}"
    )
    print(header)
    print("-" * len(header))
    for case in cases:
        best = case.best
        fb = best.get("fidelity_baseline")
        fa = best.get("fidelity_audited")
        df = best.get("fidelity_gain_pct")
        src = "vf2" if case.layout_source.startswith("vf2") else "sabre+post"
        print(
            f"{case.name:<18}{src:<14}"
            f"{best['esp_baseline']:>9.4f}{best['esp_audited']:>9.4f}"
            f"{best['esp_gain_pct']:>+8.2f}"
            f"{(f'{fb:.4f}' if fb is not None else '--'):>9}"
            f"{(f'{fa:.4f}' if fa is not None else '--'):>9}"
            f"{(f'{df:+.2f}' if df is not None else '--'):>8}"
            f"{best['swaps_baseline']:>4d}/{best['swaps_audited']:<3d}"
            f"{best['depth_baseline']:>7d}/{best['depth_audited']:<5d}"
            f"{('yes' if case.sharp else 'NO'):>7}"
        )
    print()
    print("paired per-seed deltas (95% CI, n = seeds):")
    for case in cases:
        paired = case.paired
        line = (
            f"  {case.name:<18} dESP {paired['esp_mean_delta']:+.5f} "
            f"[{paired['esp_ci95'][0]:+.5f}, {paired['esp_ci95'][1]:+.5f}]  "
            f"wins {paired['esp_wins']}/{paired['n']}"
        )
        if "fidelity_mean_delta" in paired:
            line += (
                f"   dF {paired['fidelity_mean_delta']:+.5f} "
                f"[{paired['fidelity_ci95'][0]:+.5f}, {paired['fidelity_ci95'][1]:+.5f}]  "
                f"wins {paired['fidelity_wins']}/{paired['n']}"
                f"{'  SIG' if paired.get('fidelity_significant') else ''}"
            )
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="fake_sherbrooke")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--shots", type=int, default=16384)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    parser.add_argument("--scaling", nargs="+", default=["normalized"])
    parser.add_argument("--circuits", nargs="+", default=DEFAULT_SUITE)
    parser.add_argument("--no-simulate", action="store_true")
    parser.add_argument(
        "--control",
        action="store_true",
        help="add a control arm that relocates with Qiskit's own error map "
        "(no T1/T2 injection), isolating what the injection actually buys",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--write-golden", action="store_true")
    args = parser.parse_args()

    unknown = [c for c in args.circuits if c not in SUITE]
    if unknown:
        parser.error(f"unknown circuits {unknown}; choose from {sorted(SUITE)}")

    snapshot, backend = load_fake_backend(args.backend)
    print(
        f"backend={snapshot.backend_name} qubits={snapshot.num_qubits} "
        f"seeds={args.seeds} shots={args.shots} simulate={not args.no_simulate}"
    )

    payload: dict = {
        "backend": snapshot.backend_name,
        "calibration_captured_at": snapshot.captured_at.isoformat(),
        "seeds": args.seeds,
        "shots": args.shots,
        "runs": [],
    }

    for scaling in args.scaling:
        for alpha in args.alphas:
            cases: list[CaseResult] = []
            for name in args.circuits:
                print(f"  [{scaling} alpha={alpha}] {name} ...", flush=True)
                cases.append(
                    run_case(
                        name,
                        backend,
                        snapshot,
                        seeds=args.seeds,
                        shots=args.shots,
                        alpha=alpha,
                        scaling=scaling,
                        simulate=not args.no_simulate,
                        control=args.control,
                    )
                )
            print_summary(cases, alpha=alpha, scaling=scaling)
            if args.control:
                print_control(cases)
            payload["runs"].append(
                {
                    "alpha": alpha,
                    "scaling": scaling,
                    "cases": [asdict(c) for c in cases],
                }
            )

    if len(payload["runs"]) > 1:
        _print_sweep(payload)

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")
    if args.write_golden:
        GOLDEN_PATH.write_text(json.dumps(_golden_view(payload), indent=2))
        print(f"wrote {GOLDEN_PATH}")
    return 0


def print_control(cases: list[CaseResult]) -> None:
    """How much of the gain is the T1/T2 injection, and how much is just
    running a non-strict VF2PostLayout that the preset never runs?"""
    print()
    print("control arm (same relocation, Qiskit's own error map -- no injection):")
    header = (
        f"  {'circuit':<18}{'ESP base':>10}{'ESP ctrl':>10}{'ESP aud':>10}"
        f"{'ctrl vs base':>14}{'aud vs ctrl':>13}{'dF aud-ctrl':>14}"
    )
    print(header)
    for case in cases:
        ctrl = case.control
        if not ctrl:
            continue
        df = ctrl.get("fidelity_mean_delta_vs_control")
        sig = "  SIG" if ctrl.get("fidelity_significant_vs_control") else ""
        print(
            f"  {case.name:<18}{case.best['esp_baseline']:>10.4f}"
            f"{ctrl['esp_best']:>10.4f}{case.best['esp_audited']:>10.4f}"
            f"{(ctrl['esp_gain_pct_vs_baseline'] or 0):>+13.2f}%"
            f"{(ctrl['esp_gain_pct_audited_vs_control'] or 0):>+12.2f}%"
            f"{(f'{df:+.5f}' if df is not None else '--'):>14}{sig}"
        )


def _print_sweep(payload: dict) -> None:
    print("\n=== sweep summary: mean paired fidelity delta by (scaling, alpha) ===")
    print(f"{'scaling':<14}{'alpha':>7}{'mean dF':>12}{'mean dESP':>12}{'sig cases':>11}")
    for run in payload["runs"]:
        sharp = [c for c in run["cases"] if c["sharp"] and c["fid_baseline"]]
        if not sharp:
            continue
        mean_f = sum(c["paired"]["fidelity_mean_delta"] for c in sharp) / len(sharp)
        mean_e = sum(c["paired"]["esp_mean_delta"] for c in sharp) / len(sharp)
        sig = sum(1 for c in sharp if c["paired"].get("fidelity_significant"))
        print(
            f"{run['scaling']:<14}{run['alpha']:>7.2f}{mean_f:>+12.5f}"
            f"{mean_e:>+12.5f}{sig:>7d}/{len(sharp)}"
        )


def _golden_view(payload: dict) -> dict:
    """Trim to the numbers a regression gate should watch."""
    out: dict = {
        "backend": payload["backend"],
        "seeds": payload["seeds"],
        "shots": payload["shots"],
        "cases": {},
    }
    for run in payload["runs"]:
        for case in run["cases"]:
            key = f"{case['name']}|alpha={run['alpha']}|{run['scaling']}"
            out["cases"][key] = {
                "esp_baseline": case["best"]["esp_baseline"],
                "esp_audited": case["best"]["esp_audited"],
                "esp_gain_pct": case["best"]["esp_gain_pct"],
                "swaps_audited": case["best"]["swaps_audited"],
                "depth_audited": case["best"]["depth_audited"],
                "layout_source": case["layout_source"],
                "audited_layout": case["audited_layout"],
            }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
