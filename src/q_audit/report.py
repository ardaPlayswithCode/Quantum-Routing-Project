"""drawing the side by side table, the topology map and the json payload"""

import datetime as _dt

from pydantic import BaseModel, Field
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import SCHEMA_VERSION, __version__
from .feature_extract import connectivity_summary
from .metrics import MetricSet, unusable_instructions, worst_idle_penalty
from .topology import legend_lines, render_topology

_US = 1e6


class AuditReport(BaseModel):
    """what `q-audit run --json` prints"""

    schema_version: str
    tool_version: str
    generated_at: _dt.datetime
    circuit: dict
    backend: dict
    calibration: dict
    settings: dict
    baseline: MetricSet
    baseline_default_seed: MetricSet
    audited: MetricSet
    comparison: dict
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _pct(new, old):
    if old == 0:
        return None
    return 100.0 * (new / old - 1.0)


def _fmt_delta(baseline, audited, *, lower_is_better, fmt="{:+.1f}",
               as_percent=False, unit=""):
    if baseline is None or audited is None:
        return Text("--", style="grey50")
    if as_percent:
        pct = _pct(audited, baseline)
        if pct is None:
            # a zero baseline isn't "no change", it means the baseline circuit
            # can't succeed at all, so anything positive is an infinite gain.
            # printing 0.00% here would be a lie
            if audited > 0:
                return Text("from zero", style="bold green")
            return Text("--", style="grey50")
        value = pct
        suffix = "%"
    else:
        value = audited - baseline
        suffix = unit
    if abs(value) < 1e-12:
        return Text("same", style="grey50")
    better = (value < 0) if lower_is_better else (value > 0)
    arrow = "▼" if value < 0 else "▲"
    return Text(f"{arrow} {fmt.format(value)}{suffix}", style="green" if better else "red")


def _row(table, label, baseline, audited, getter, *, fmt="{:.0f}",
         lower_is_better=True, scale=1.0, as_percent=False, unit="", emphasise=False):
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


def build_table(result, snapshot):
    b = result.baseline.best
    a = result.audited.best
    n = result.settings.seeds
    table = Table(
        title="Transpilation audit: baseline vs T1/T2-aware layout",
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


def build_header(result, snapshot, features, origin):
    age_h = snapshot.age_hours()
    age = f"{age_h:.1f}h" if age_h < 48 else f"{age_h / 24:.0f}d"
    lines = [
        f"[bold]circuit[/]      {features.name}  ({origin})",
        f"[bold]shape[/]        {features.num_qubits}q, depth {features.depth}, "
        f"{features.two_qubit_gates} two-qubit gates, {connectivity_summary(features, snapshot)}",
        f"[bold]backend[/]      {snapshot.backend_name}  ({snapshot.num_qubits} qubits, "
        f"{len(snapshot.coupling)} couplers)",
        f"[bold]calibration[/]  source={snapshot.source}  "
        f"captured={snapshot.captured_at:%Y-%m-%d %H:%M %Z}  age={age}",
        f"[bold]settings[/]     alpha={result.settings.alpha}  seeds={result.settings.seeds} "
        f"(base {result.settings.base_seed})  opt_level={result.settings.optimization_level}  "
        f"schedule={result.settings.schedule_method}  map={result.error_map.scaling}",
        f"[bold]strategy[/]     {result.audited_strategy}  "
        f"(t_est={result.error_map.t_est_s * _US:.2f}us, "
        f"vf2={result.layout_info.get('vf2_stop_reason', '?').split('.')[-1]})",
    ]
    return Panel("\n".join(lines), title="q-audit", title_align="left", border_style="cyan")


def build_idle_table(result, snapshot):
    table = Table(
        title="Worst idle windows (audited circuit), ie where coherence is going",
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


def render_terminal(result, snapshot, features, origin, console, *, show_topology=True):
    console.print(build_header(result, snapshot, features, origin))
    console.print(build_table(result, snapshot))

    b = result.baseline.best
    a = result.audited.best
    console.print(
        f"\n[bold]baseline layout[/] (seed {b.seed}): "
        + " ".join(str(q) for q in b.layout)
    )
    same = "   [grey62](identical to baseline)[/]" if a.layout == b.layout else ""
    console.print(
        f"[bold]audited  layout[/] (seed {a.seed}): "
        + " ".join(str(q) for q in a.layout)
        + same
    )

    for label, path in (("baseline", result.baseline), ("audited", result.audited)):
        dead = unusable_instructions(path.best_circuit, snapshot)
        if dead:
            listing = ", ".join(
                f"{d['instruction']}{tuple(d['qubits'])} err={d['error']:.3f}"
                for d in dead[:4]
            )
            extra = f" (+{len(dead) - 4} more)" if len(dead) > 4 else ""
            console.print(f"\n[bold red]DEAD HARDWARE on the {label} layout:[/] {listing}{extra}")
            console.print(
                "[red]calibration says these are unusable, this circuit cannot "
                "succeed as routed.[/]"
            )

    if result.recommendation == "adopt_audited":
        gain = _pct(a.esp, b.esp)
        change = f"{gain:+.2f}%" if gain is not None else "baseline ESP was zero"
        console.print(
            f"\n[bold green]VERDICT: adopt the audited layout[/] "
            f"- ESP {b.esp:.4f} -> {a.esp:.4f} ({change}) via {result.audited_strategy}"
        )
    else:
        console.print(
            "\n[bold yellow]VERDICT: keep the baseline[/] - the T1/T2-aware layout "
            f"didn't improve ESP here ({b.esp:.4f} vs {a.esp:.4f})."
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
    if default.esp != result.baseline.best.esp:
        console.print(
            f"\n[grey62]stock single-seed baseline (seed {default.seed}): "
            f"ESP {default.esp:.4f}. the table's baseline is the best of "
            f"{result.settings.seeds} seeds ({result.baseline.best.esp:.4f}) so both "
            f"paths get the same sampling budget.[/]"
        )

    unknown = set(result.baseline.best_schedule.unknown_durations) | set(
        result.audited.best_schedule.unknown_durations
    )
    if unknown:
        console.print(
            f"[yellow]! no duration in the target for {sorted(unknown)}, those "
            "instructions were treated as instant so duration and idle time are "
            "underestimates.[/]"
        )

    for warning in result.warnings:
        console.print(f"[yellow]! {warning}[/]")
    for note in result.notes + features.notes:
        console.print(f"[grey62]. {note}[/]")


def comparison_dict(result, snapshot=None):
    b = result.baseline.best
    a = result.audited.best
    dead = {}
    if snapshot is not None:
        dead = {
            "baseline_unusable_instructions": unusable_instructions(
                result.baseline.best_circuit, snapshot
            ),
            "audited_unusable_instructions": unusable_instructions(
                result.audited.best_circuit, snapshot
            ),
        }
    if a.worst_t2_on_layout_s is None or b.worst_t2_on_layout_s is None:
        worst_t2_delta = None
    else:
        worst_t2_delta = a.worst_t2_on_layout_s - b.worst_t2_on_layout_s
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
        "worst_t2_delta_s": worst_t2_delta,
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


def build_json_report(result, snapshot, features, origin, warnings):
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
