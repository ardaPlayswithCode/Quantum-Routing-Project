"""Rendering: the rich side-by-side table, the topology map, and the JSON payload."""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import SCHEMA_VERSION, __version__
from .feature_extract import CircuitFeatures, connectivity_summary
from .metrics import unusable_instructions, worst_idle_penalty
from .models import AuditReport, CalibrationSnapshot, MetricSet
from .topology import legend_lines, render_topology
from .transpile_audit import AuditResult

_US = 1e6


def _pct(new: float, old: float) -> float | None:
    if old == 0:
        return None
    return 100.0 * (new / old - 1.0)


def _fmt_delta(
    baseline: float | None,
    audited: float | None,
    *,
    lower_is_better: bool,
    fmt: str = "{:+.1f}",
    as_percent: bool = False,
    unit: str = "",
) -> Text:
    if baseline is None or audited is None:
        return Text("--", style="grey50")
    if as_percent:
        pct = _pct(audited, baseline)
        if pct is None:
            # A zero baseline is not "no change"; it means the baseline circuit
            # cannot succeed at all, so any positive audited value is an
            # infinite relative improvement. Saying "0.00%" would be a lie.
            if audited > 0:
                return Text("from zero", style="bold green")
            return Text("--", style="grey50")
        value, suffix = pct, "%"
    else:
        value, suffix = audited - baseline, unit
    if abs(value) < 1e-12:
        return Text("same", style="grey50")
    better = (value < 0) if lower_is_better else (value > 0)
    arrow = "▼" if value < 0 else "▲"
    return Text(f"{arrow} {fmt.format(value)}{suffix}", style="green" if better else "red")


def _row(
    table: Table,
    label: str,
    baseline: MetricSet,
    audited: MetricSet,
    getter: Callable[[MetricSet], float | None],
    *,
    fmt: str = "{:.0f}",
    lower_is_better: bool = True,
    scale: float = 1.0,
    as_percent: bool = False,
    unit: str = "",
    emphasise: bool = False,
) -> None:
    b = getter(baseline)
    a = getter(audited)
    bs = "--" if b is None else fmt.format(b * scale)
    as_ = "--" if a is None else fmt.format(a * scale)
    delta = _fmt_delta(
        None if b is None else b * scale,
        None if a is None else a * scale,
        lower_is_better=lower_is_better,
        fmt="{:+.2f}" if as_percent else "{:+.4g}",
        as_percent=as_percent,
        unit=unit,
    )
    style = "bold" if emphasise else None
    table.add_row(Text(label, style=style), Text(bs, style=style), Text(as_, style=style), delta)


def build_table(result: AuditResult, snapshot: CalibrationSnapshot) -> Table:
    b, a = result.baseline.best, result.audited.best
    n = result.settings.seeds
    table = Table(
        title="Transpilation audit — baseline vs T1/T2-aware layout",
        title_style="bold",
        header_style="bold cyan",
        show_lines=False,
        expand=False,
    )
    table.add_column("Metric", style="white", no_wrap=True)
    table.add_column(f"Baseline\n(best of {n})", justify="right")
    table.add_column(f"Audited\n(best of {n})", justify="right")
    table.add_column("Change", justify="right")

    _row(table, "ESP (estimated success prob.)", b, a, lambda m: m.esp,
         fmt="{:.4f}", lower_is_better=False, as_percent=True, emphasise=True)
    _row(table, "  ... gate-error component", b, a, lambda m: m.esp_gate_only,
         fmt="{:.4f}", lower_is_better=False, as_percent=True)
    _row(table, "  ... idle-decoherence component", b, a, lambda m: m.esp_idle_only,
         fmt="{:.4f}", lower_is_better=False, as_percent=True)
    table.add_section()
    _row(table, "Circuit depth", b, a, lambda m: m.depth, fmt="{:.0f}")
    _row(table, "Total instructions", b, a, lambda m: m.size, fmt="{:.0f}")
    _row(table, "Two-qubit gates", b, a, lambda m: m.two_qubit_gates, fmt="{:.0f}")
    _row(table, "Routing SWAPs", b, a, lambda m: m.swap_gates, fmt="{:.0f}")
    table.add_section()
    _row(table, "Duration (us)", b, a, lambda m: m.duration_s, fmt="{:.2f}", scale=_US)
    _row(table, "Total qubit-idle time (us)", b, a, lambda m: m.total_idle_s,
         fmt="{:.2f}", scale=_US)
    table.add_section()
    _row(table, "Worst T2 on layout (us)", b, a, lambda m: m.worst_t2_on_layout_s,
         fmt="{:.1f}", scale=_US, lower_is_better=False)
    _row(table, "Mean T2 on layout (us)", b, a, lambda m: m.mean_t2_on_layout_s,
         fmt="{:.1f}", scale=_US, lower_is_better=False)
    _row(table, "Worst T1 on layout (us)", b, a, lambda m: m.worst_t1_on_layout_s,
         fmt="{:.1f}", scale=_US, lower_is_better=False)
    _row(table, "Worst 2q edge error", b, a, lambda m: m.worst_edge_error, fmt="{:.5f}")
    _row(table, "Readout error (sum over layout)", b, a, lambda m: m.readout_error_sum,
         fmt="{:.4f}")
    if b.hellinger_fidelity is not None or a.hellinger_fidelity is not None:
        table.add_section()
        _row(table, "Hellinger fidelity (Aer)", b, a, lambda m: m.hellinger_fidelity,
             fmt="{:.4f}", lower_is_better=False, as_percent=True, emphasise=True)
    return table


def build_header(
    result: AuditResult,
    snapshot: CalibrationSnapshot,
    features: CircuitFeatures,
    origin: str,
) -> Panel:
    age_h = snapshot.age_hours()
    age = f"{age_h:.1f}h" if age_h < 48 else f"{age_h / 24:.0f}d"
    lines = [
        f"[bold]circuit[/]      {features.name}  ({origin})",
        f"[bold]shape[/]        {features.num_qubits}q, depth {features.depth}, "
        f"{features.two_qubit_gates} two-qubit gates, {connectivity_summary(features, snapshot)}",
        f"[bold]backend[/]      {snapshot.backend_name}  ({snapshot.num_qubits} qubits, "
        f"{len(snapshot.coupling)} couplers)",
        f"[bold]calibration[/]  source={snapshot.source}  captured={snapshot.captured_at:%Y-%m-%d %H:%M %Z}  age={age}",
        f"[bold]settings[/]     alpha={result.settings.alpha}  seeds={result.settings.seeds} "
        f"(base {result.settings.base_seed})  opt_level={result.settings.optimization_level}  "
        f"schedule={result.settings.schedule_method}  map={result.error_map.scaling}",
        f"[bold]strategy[/]     {result.audited_strategy}  "
        f"(t_est={result.error_map.t_est_s * _US:.2f}us, "
        f"vf2={result.layout_info.get('vf2_stop_reason', '?').split('.')[-1]})",
    ]
    return Panel("\n".join(lines), title="q-audit", title_align="left", border_style="cyan")


def build_idle_table(result: AuditResult, snapshot: CalibrationSnapshot) -> Table:
    table = Table(
        title="Worst idle windows (audited circuit) — where coherence is being spent",
        title_style="bold",
        header_style="bold cyan",
    )
    table.add_column("Qubit", justify="right")
    table.add_column("Idle (us)", justify="right")
    table.add_column("T1 (us)", justify="right")
    table.add_column("T2 (us)", justify="right")
    table.add_column("Infidelity", justify="right")
    for row in worst_idle_penalty(result.audited.best_schedule, snapshot, top=5):
        table.add_row(
            str(row["qubit"]),
            f"{row['idle_s'] * _US:.2f}",
            "--" if row["t1_s"] is None else f"{row['t1_s'] * _US:.0f}",
            "--" if row["t2_s"] is None else f"{row['t2_s'] * _US:.0f}",
            f"{row['infidelity']:.4f}",
        )
    return table


def render_terminal(
    result: AuditResult,
    snapshot: CalibrationSnapshot,
    features: CircuitFeatures,
    origin: str,
    console: Console,
    *,
    show_topology: bool = True,
) -> None:
    console.print(build_header(result, snapshot, features, origin))
    console.print(build_table(result, snapshot))

    b, a = result.baseline.best, result.audited.best
    console.print(
        f"\n[bold]baseline layout[/] (seed {b.seed}): "
        + " ".join(str(q) for q in b.layout)
    )
    console.print(
        f"[bold]audited  layout[/] (seed {a.seed}): "
        + " ".join(str(q) for q in a.layout)
        + ("   [grey62](identical to baseline)[/]" if a.layout == b.layout else "")
    )

    for label, path in (("baseline", result.baseline), ("audited", result.audited)):
        dead = unusable_instructions(path.best_circuit, snapshot)
        if dead:
            listing = ", ".join(
                f"{d['instruction']}{tuple(d['qubits'])} err={d['error']:.3f}"
                for d in dead[:4]
            )
            console.print(
                f"\n[bold red]DEAD HARDWARE on the {label} layout:[/] {listing}"
                + (f" (+{len(dead) - 4} more)" if len(dead) > 4 else "")
            )
            console.print(
                "[red]The calibration reports these as unusable; this circuit "
                "cannot succeed as routed.[/]"
            )

    verdict = result.recommendation
    if verdict == "adopt_audited":
        gain = _pct(a.esp, b.esp)
        change = f"{gain:+.2f}%" if gain is not None else "baseline ESP was zero"
        console.print(
            f"\n[bold green]VERDICT: adopt the audited layout[/] "
            f"— ESP {b.esp:.4f} -> {a.esp:.4f} ({change}) via {result.audited_strategy}"
        )
    else:
        console.print(
            "\n[bold yellow]VERDICT: keep the baseline[/] — the T1/T2-aware layout "
            f"did not improve ESP here ({b.esp:.4f} vs {a.esp:.4f})."
        )
    if len(result.strategies) > 1:
        detail = "  ".join(
            f"{name}={path.best.esp:.4f}" for name, path in result.strategies.items()
        )
        console.print(f"[grey62]audited strategies tried: {detail}[/]")

    if show_topology:
        console.print()
        console.print(Text("Device topology (colour = T2 tier)", style="bold"))
        for line in render_topology(
            snapshot,
            baseline_layout=result.baseline.best.layout,
            audited_layout=result.audited.best.layout,
        ):
            console.print(line, highlight=False)
        for line in legend_lines(snapshot):
            console.print(line, highlight=False)

    console.print()
    console.print(build_idle_table(result, snapshot))

    default = result.baseline_default
    best = result.baseline.best
    if default.esp != best.esp:
        console.print(
            f"\n[grey62]Stock single-seed baseline (seed {default.seed}): "
            f"ESP {default.esp:.4f}; the table's baseline is the best of "
            f"{result.settings.seeds} seeds ({best.esp:.4f}) so both paths get the "
            f"same sampling budget.[/]"
        )

    unknown = set(result.baseline.best_schedule.unknown_durations) | set(
        result.audited.best_schedule.unknown_durations
    )
    if unknown:
        console.print(
            f"[yellow]! No duration in the target for {sorted(unknown)}; those "
            "instructions were treated as instantaneous, so duration and idle "
            "time are underestimates.[/]"
        )

    for warning in result.warnings:
        console.print(f"[yellow]! {warning}[/]")
    for note in result.notes + features.notes:
        console.print(f"[grey62]. {note}[/]")


def comparison_dict(result: AuditResult, snapshot: CalibrationSnapshot | None = None) -> dict:
    b, a = result.baseline.best, result.audited.best
    dead: dict = {}
    if snapshot is not None:
        dead = {
            "baseline_unusable_instructions": unusable_instructions(
                result.baseline.best_circuit, snapshot
            ),
            "audited_unusable_instructions": unusable_instructions(
                result.audited.best_circuit, snapshot
            ),
        }
    return {
        **dead,
        "esp_baseline": b.esp,
        "esp_audited": a.esp,
        "esp_delta": a.esp - b.esp,
        "esp_delta_pct": _pct(a.esp, b.esp),
        "esp_idle_delta_pct": _pct(a.esp_idle_only, b.esp_idle_only),
        "esp_gate_delta_pct": _pct(a.esp_gate_only, b.esp_gate_only),
        "depth_delta": a.depth - b.depth,
        "two_qubit_delta": a.two_qubit_gates - b.two_qubit_gates,
        "swap_delta": a.swap_gates - b.swap_gates,
        "duration_delta_s": a.duration_s - b.duration_s,
        "idle_delta_s": a.total_idle_s - b.total_idle_s,
        "worst_t2_delta_s": (
            None
            if a.worst_t2_on_layout_s is None or b.worst_t2_on_layout_s is None
            else a.worst_t2_on_layout_s - b.worst_t2_on_layout_s
        ),
        "audited_wins": a.esp > b.esp,
        "recommendation": result.recommendation,
        "audited_strategy": result.audited_strategy,
        "strategy_esp": {name: path.best.esp for name, path in result.strategies.items()},
        "layouts_identical": a.layout == b.layout,
        "esp_series_baseline": result.baseline.esp_series(),
        "esp_series_audited": result.audited.esp_series(),
        "unknown_durations": sorted(
            set(result.baseline.best_schedule.unknown_durations)
            | set(result.audited.best_schedule.unknown_durations)
        ),
    }


def build_json_report(
    result: AuditResult,
    snapshot: CalibrationSnapshot,
    features: CircuitFeatures,
    origin: str,
    warnings: list[str],
) -> AuditReport:
    return AuditReport(
        schema_version=SCHEMA_VERSION,
        tool_version=__version__,
        generated_at=_dt.datetime.now(_dt.timezone.utc),
        circuit={**features.to_dict(), "origin": origin},
        backend={
            "name": snapshot.backend_name,
            "num_qubits": snapshot.num_qubits,
            "num_couplers": len(snapshot.coupling),
            "basis_gates": snapshot.basis_gates,
            "dt": snapshot.dt,
        },
        calibration={
            "source": snapshot.source,
            "captured_at": snapshot.captured_at.isoformat(),
            "age_hours": snapshot.age_hours(),
            "missing_fields": len(snapshot.missing_fields()),
            "median_two_qubit_duration_s": snapshot.median_two_qubit_duration(),
        },
        settings={
            "alpha": result.settings.alpha,
            "seeds": result.settings.seeds,
            "base_seed": result.settings.base_seed,
            "seed_list": result.settings.seed_list(),
            "optimization_level": result.settings.optimization_level,
            "schedule_method": result.settings.schedule_method,
            "map_scaling": result.error_map.scaling,
            "error_map": result.error_map.to_dict(),
            "layout": result.layout_info,
        },
        baseline=result.baseline.best,
        baseline_default_seed=result.baseline_default,
        audited=result.audited.best,
        comparison=comparison_dict(result, snapshot),
        warnings=warnings + result.warnings,
        notes=result.notes + features.notes,
    )
