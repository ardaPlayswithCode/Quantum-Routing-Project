"""``q-audit`` command line interface.

Output contract
---------------
* stdout carries the report, and with ``--json`` carries exactly one JSON
  document -- including on failure, so ``q-audit ... --json | jq`` never sees
  a truncated stream.
* stderr carries NDJSON progress events and human-readable errors.
* exit codes: 0 ok, 2 user error, 3 calibration error, 4 transpiler error.
"""

from __future__ import annotations

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
from .progress import ProgressReporter
from .report import build_json_report, render_terminal
from .transpile_audit import AuditSettings, run_audit

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Audit a quantum circuit's transpilation: Qiskit defaults vs a "
    "T1/T2-aware layout, side by side.",
)


def _fail(exc: QAuditError, *, as_json: bool, console: Console) -> None:
    """Emit a failure the caller can always parse, then exit with its code."""
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
        typer.Option(
            "--map-scaling",
            help="Diagonal scaling for the injected map: 'normalized' (corrects for "
            "VF2's per-op exponent) or 'raw'.",
        ),
    ] = "normalized",
    no_post_layout: Annotated[
        bool, typer.Option("--no-post-layout", help="Skip the VF2PostLayout refinement rounds.")
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
                     "Hellinger fidelity (needs qiskit-aer; slow)."),
    ] = False,
    shots: Annotated[int, typer.Option("--shots", help="Shots for --verify.")] = 8192,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Silence stderr progress.")] = False,
) -> None:
    """Transpile CIRCUIT twice and compare the results side by side."""
    console = Console(stderr=False)
    err_console = Console(stderr=True)
    reporter = ProgressReporter(quiet=quiet)

    try:
        _validate_choices(
            alpha=alpha, seeds=seeds, schedule=schedule, map_scaling=map_scaling
        )
        if not error_map_is_supported():
            raise TranspileAuditError(
                "This Qiskit build no longer exposes the VF2 'vf2_avg_error_map' hook.",
                hint="q-audit pins qiskit==2.5.1 for exactly this reason; check your env.",
            )

        reporter.step("load_circuit", source=circuit)
        qc, origin = load_circuit(circuit)

        reporter.step("resolve_calibration", backend=backend)
        snapshot, backend_obj = resolve_calibration(backend, refresh=refresh)
        warnings = check_staleness(snapshot, allow_stale=allow_stale)
        for warning in warnings:
            reporter.warn(warning)

        reporter.step("extract_features")
        bound, unrolled, features = extract_features(
            qc, snapshot, reject_control_flow=not allow_control_flow
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
            bound, unrolled, features, snapshot, backend_obj, settings, reporter=reporter
        )

        if verify:
            _attach_verification(result, backend_obj, bound, shots, reporter)

        if as_json:
            report = build_json_report(result, snapshot, features, origin, warnings)
            sys.stdout.write(report.model_dump_json(indent=2) + "\n")
        else:
            result.warnings.extend(warnings)
            render_terminal(
                result, snapshot, features, origin, console, show_topology=topology
            )
        reporter.step("done")
    except QAuditError as exc:
        reporter.error(exc.message)
        _fail(exc, as_json=as_json, console=err_console)
    except KeyboardInterrupt:  # pragma: no cover
        raise typer.Exit(code=130) from None
    except Exception as exc:  # noqa: BLE001 - last-resort guard keeps the contract
        reporter.error(f"{type(exc).__name__}: {exc}")
        wrapped = TranspileAuditError(
            f"Unexpected {type(exc).__name__}: {exc}",
            hint="This is a bug in q-audit or Qiskit API drift; re-run with --quiet off "
            "to see the progress trail.",
        )
        _fail(wrapped, as_json=as_json, console=err_console)


def _validate_choices(*, alpha: float, seeds: int, schedule: str, map_scaling: str) -> None:
    """Turn bad flags into exit-2 user errors, not exit-4 "unexpected" crashes."""
    from .errors import QAuditError as _E

    if alpha < 0:
        raise _E("--alpha must be >= 0.", hint="0 disables the idle-decoherence term.")
    if seeds < 1:
        raise _E("--seeds must be >= 1.")
    if schedule.lower() not in ("asap", "alap"):
        raise _E(f"--schedule must be 'asap' or 'alap', got {schedule!r}.")
    if map_scaling.lower() not in ("raw", "normalized"):
        raise _E(
            f"--map-scaling must be 'normalized' or 'raw', got {map_scaling!r}.",
            hint="'normalized' corrects for VF2 exponentiating the diagonal by "
            "the per-qubit 1q op count.",
        )


def _attach_verification(result, backend_obj, bound, shots: int, reporter) -> None:
    """Run the Aer harness and fold Hellinger fidelity into both MetricSets."""
    from .verify import ensure_measured, ideal_distribution, require_aer, simulated_fidelity

    require_aer()  # clean, typed error instead of a raw ImportError
    from qiskit_aer.noise import NoiseModel

    reporter.step("verify_start", shots=shots)
    ideal = ideal_distribution(ensure_measured(bound))
    # One NoiseModel, shared: building it from a 127-qubit backend is the
    # expensive part, and both paths must be judged by the same noise.
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
        snapshot, _ = resolve_calibration(backend)
        warnings = check_staleness(snapshot, allow_stale=allow_stale)
        if as_json:
            payload = json.loads(snapshot.model_dump_json())
            payload["age_hours"] = snapshot.age_hours()
            payload["warnings"] = warnings
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
            return
        t2 = sorted(q.t2 for q in snapshot.qubits if q.t2 is not None)
        console.print(f"[bold]{snapshot.backend_name}[/]  source={snapshot.source}")
        console.print(f"captured {snapshot.captured_at}  (age {snapshot.age_hours():.1f}h)")
        console.print(
            f"{snapshot.num_qubits} qubits, {len(snapshot.coupling)} couplers, "
            f"{len(snapshot.edges)} calibrated 2q directions"
        )
        if t2:
            console.print(
                f"T2 min/median/max: {t2[0] * 1e6:.1f} / {t2[len(t2) // 2] * 1e6:.1f} / "
                f"{t2[-1] * 1e6:.1f} us"
            )
        console.print(f"missing calibration fields: {len(snapshot.missing_fields())}")
        for warning in warnings:
            console.print(f"[yellow]! {warning}[/]")
    except QAuditError as exc:
        _fail(exc, as_json=as_json, console=err_console)


@app.command()
def version() -> None:
    """Print version and environment information."""
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
