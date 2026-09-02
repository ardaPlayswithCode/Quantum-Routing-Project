"""the command line interface.

output rules, which the tests check: stdout gets the report, and with --json it
gets exactly one json document and nothing else, even when things fail, so that
`q-audit ... --json | jq` never sees half a stream. stderr gets the progress
lines and any error. exit codes are 0 ok, 2 user error, 3 calibration, 4
transpiler
"""

import json
import sys
from typing import Annotated

import typer
from rich.console import Console

from . import SCHEMA_VERSION, __version__
from .calibration import check_staleness, resolve_calibration
from .circuit_loader import load_circuit
from .errors import QAuditError, TranspileAuditError
from .feature_extract import extract_features
from .passes import error_map_is_supported
from .report import build_json_report, render_terminal
from .transpile_audit import AuditSettings, ProgressReporter, run_audit

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Audit a circuit's transpilation: qiskit defaults vs a T1/T2-aware layout.",
)


def _fail(exc, *, as_json, console):
    """print the failure in whichever format the caller asked for, then exit"""
    if as_json:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "tool_version": __version__,
            "ok": False,
            "error": exc.to_dict(),
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        sys.stdout.flush()
    else:
        console.print(f"[bold red]error:[/] {exc.message}")
        if exc.hint:
            console.print(f"[grey62]hint:[/] {exc.hint}")
    raise typer.Exit(code=exc.exit_code)


@app.command()
def run(
    circuit: Annotated[
        str,
        typer.Argument(help="Circuit file (.py defining `qc`, or .qasm), or '-' for QASM on stdin."),
    ],
    backend: Annotated[
        str, typer.Option("--backend", "-b", help="Backend name, e.g. fake_sherbrooke.")
    ] = "fake_sherbrooke",
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit one JSON document on stdout instead of a table.")
    ] = False,
    allow_stale: Annotated[
        bool, typer.Option("--allow-stale", help="Proceed even if calibration is over 24h old.")
    ] = False,
    alpha: Annotated[
        float, typer.Option("--alpha", help="Weight on the idle-decoherence penalty.")
    ] = 1.0,
    seeds: Annotated[
        int, typer.Option("--seeds", help="Transpiler seeds per path (both paths get the same list).")
    ] = 20,
    base_seed: Annotated[int, typer.Option("--base-seed", help="First seed in the list.")] = 42,
    optimization_level: Annotated[
        int, typer.Option("--optimization-level", "-O", min=0, max=3)
    ] = 3,
    schedule: Annotated[
        str, typer.Option("--schedule", help="Scheduling for idle analysis: asap or alap.")
    ] = "asap",
    map_scaling: Annotated[
        str,
        typer.Option("--map-scaling", help="Diagonal scaling of the injected map: normalized or raw."),
    ] = "normalized",
    no_post_layout: Annotated[
        bool, typer.Option("--no-post-layout", help="Skip the VF2PostLayout relocate strategy.")
    ] = False,
    topology: Annotated[
        bool, typer.Option("--topology/--no-topology", help="Draw the ASCII device map.")
    ] = True,
    allow_control_flow: Annotated[
        bool, typer.Option("--allow-control-flow", help="Audit circuits with if/for/switch anyway.")
    ] = False,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Ignore the calibration cache and refetch.")
    ] = False,
    verify: Annotated[
        bool,
        typer.Option("--verify", help="Also simulate both circuits with Aer and report "
                     "Hellinger fidelity (needs qiskit-aer, slow)."),
    ] = False,
    shots: Annotated[int, typer.Option("--shots", help="Shots for --verify.")] = 8192,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Silence stderr progress.")] = False,
) -> None:
    """Transpile CIRCUIT twice and compare the results side by side."""
    console = Console(stderr=False)
    err_console = Console(stderr=True)
    reporter = ProgressReporter(quiet=quiet)

    try:
        _validate_choices(alpha=alpha, seeds=seeds, schedule=schedule, map_scaling=map_scaling)
        if not error_map_is_supported():
            raise TranspileAuditError(
                "this qiskit build doesn't expose the vf2 'vf2_avg_error_map' hook anymore",
                hint="we pin qiskit==2.5.1 for exactly this reason, check your env",
            )

        reporter.step("load_circuit", source=circuit)
        qc, origin = load_circuit(circuit)

        reporter.step("resolve_calibration", backend=backend)
        snap, backend_obj = resolve_calibration(backend, refresh=refresh)
        warnings = check_staleness(snap, allow_stale=allow_stale)
        for warning in warnings:
            reporter.warn(warning)

        reporter.step("extract_features")
        bound, unrolled, features = extract_features(
            qc, snap, reject_control_flow=not allow_control_flow
        )

        settings = AuditSettings(
            backend=backend,
            alpha=alpha,
            seeds=seeds,
            base_seed=base_seed,
            optimization_level=optimization_level,
            schedule_method=schedule,
            post_layout_refine=not no_post_layout,
            map_scaling=map_scaling,
        )
        result = run_audit(
            bound, unrolled, features, snap, backend_obj, settings, reporter=reporter
        )

        if verify:
            _attach_verification(result, backend_obj, bound, shots, reporter)

        if as_json:
            report = build_json_report(result, snap, features, origin, warnings)
            sys.stdout.write(report.model_dump_json(indent=2) + "\n")
        else:
            result.warnings.extend(warnings)
            render_terminal(result, snap, features, origin, console, show_topology=topology)
        reporter.step("done")
    except QAuditError as exc:
        reporter.error(exc.message)
        _fail(exc, as_json=as_json, console=err_console)
    except KeyboardInterrupt:
        raise typer.Exit(code=130) from None
    except Exception as exc:
        # catch all so --json still gets one document out no matter what
        reporter.error(f"{type(exc).__name__}: {exc}")
        wrapped = TranspileAuditError(
            f"unexpected {type(exc).__name__}: {exc}",
            hint="that's a bug in q-audit, or qiskit changed under us. re-run without "
            "--quiet to see the progress trail.",
        )
        _fail(wrapped, as_json=as_json, console=err_console)


def _validate_choices(*, alpha, seeds, schedule, map_scaling):
    """turn bad flags into exit-2 user errors instead of exit-4 crashes"""
    if alpha < 0:
        raise QAuditError("--alpha can't be negative", hint="0 turns the idle term off")
    if seeds < 1:
        raise QAuditError("--seeds must be at least 1")
    if schedule.lower() not in ("asap", "alap"):
        raise QAuditError(f"--schedule has to be 'asap' or 'alap', got {schedule!r}")
    if map_scaling.lower() not in ("raw", "normalized"):
        raise QAuditError(
            f"--map-scaling has to be 'normalized' or 'raw', got {map_scaling!r}",
            hint="normalized corrects for vf2 exponentiating the diagonal by the "
            "per qubit 1q op count",
        )


def _attach_verification(result, backend_obj, bound, shots, reporter):
    """run the aer harness and stick the fidelity onto both metric sets"""
    from .verify import ensure_measured, ideal_distribution, require_aer, simulated_fidelity

    require_aer()  # so you get a proper error instead of a raw ImportError
    from qiskit_aer.noise import NoiseModel

    reporter.step("verify_start", shots=shots)
    ideal = ideal_distribution(ensure_measured(bound))
    # build the noise model once. it's the expensive part on a 127 qubit
    # backend and both paths have to be judged by the same noise anyway
    noise_model = NoiseModel.from_backend(backend_obj)
    for path in (result.baseline, result.audited):
        fidelity, _counts = simulated_fidelity(
            path.best_circuit,
            ideal,
            backend_obj,
            shots=shots,
            seed_simulator=result.settings.base_seed,
            schedule_method=result.settings.schedule_method,
            noise_model=noise_model,
        )
        path.best = path.best.model_copy(update={"hellinger_fidelity": fidelity})
        reporter.step("verify", path=path.label, hellinger=round(fidelity, 5))
    if result.baseline_default.seed == result.baseline.best.seed:
        result.baseline_default = result.baseline_default.model_copy(
            update={"hellinger_fidelity": result.baseline.best.hellinger_fidelity}
        )


@app.command()
def calibration(
    backend: Annotated[str, typer.Argument(help="Backend name.")] = "fake_sherbrooke",
    as_json: Annotated[bool, typer.Option("--json")] = False,
    allow_stale: Annotated[bool, typer.Option("--allow-stale")] = False,
) -> None:
    """Show the calibration snapshot q-audit would use."""
    console = Console()
    err_console = Console(stderr=True)
    try:
        snap, _ = resolve_calibration(backend)
        warnings = check_staleness(snap, allow_stale=allow_stale)
        if as_json:
            payload = json.loads(snap.model_dump_json())
            payload["age_hours"] = snap.age_hours()
            payload["warnings"] = warnings
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
            return
        t2 = sorted(q.t2 for q in snap.qubits if q.t2 is not None)
        console.print(f"[bold]{snap.backend_name}[/]  source={snap.source}")
        console.print(f"captured {snap.captured_at}  (age {snap.age_hours():.1f}h)")
        console.print(
            f"{snap.num_qubits} qubits, {len(snap.coupling)} couplers, "
            f"{len(snap.edges)} calibrated 2q directions"
        )
        if t2:
            console.print(
                f"T2 min/median/max: {t2[0] * 1e6:.1f} / {t2[len(t2) // 2] * 1e6:.1f} / "
                f"{t2[-1] * 1e6:.1f} us"
            )
        console.print(f"missing calibration fields: {len(snap.missing_fields())}")
        for warning in warnings:
            console.print(f"[yellow]! {warning}[/]")
    except QAuditError as exc:
        _fail(exc, as_json=as_json, console=err_console)


@app.command()
def version() -> None:
    """Print version and environment info."""
    import qiskit

    typer.echo(f"q-audit {__version__} (schema {SCHEMA_VERSION})")
    typer.echo(f"qiskit {qiskit.__version__}")
    try:
        import qiskit_ibm_runtime

        typer.echo(f"qiskit-ibm-runtime {qiskit_ibm_runtime.__version__}")
    except ImportError:
        typer.echo("qiskit-ibm-runtime not installed")
    try:
        import qiskit_aer

        typer.echo(f"qiskit-aer {qiskit_aer.__version__} (dev only)")
    except ImportError:
        typer.echo("qiskit-aer not installed (--verify unavailable)")
    typer.echo(f"vf2 error-map hook: {'ok' if error_map_is_supported() else 'MISSING'}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
