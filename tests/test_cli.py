"""CLI contract: stdout purity, JSON validity (including on failure), exit codes."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from q_audit import SCHEMA_VERSION
from q_audit.cli import app

runner = CliRunner()

FAST = ["--seeds", "2", "--no-topology"]


@pytest.fixture(scope="module")
def json_result(examples_dir):
    return runner.invoke(
        app, ["run", str(examples_dir / "ghz7.py"), "--json", *FAST]
    )


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "q-audit" in result.stdout
    assert "vf2 error-map hook: ok" in result.stdout


def test_json_run_exits_clean(json_result):
    assert json_result.exit_code == 0, json_result.stdout


def test_json_stdout_is_exactly_one_document(json_result):
    payload = json.loads(json_result.stdout)  # raises if stdout is polluted
    assert isinstance(payload, dict)


def test_json_has_the_documented_schema(json_result):
    payload = json.loads(json_result.stdout)
    assert payload["schema_version"] == SCHEMA_VERSION
    for key in (
        "tool_version",
        "generated_at",
        "circuit",
        "backend",
        "calibration",
        "settings",
        "baseline",
        "baseline_default_seed",
        "audited",
        "comparison",
        "warnings",
        "notes",
    ):
        assert key in payload, f"missing top-level key {key}"

    for path in ("baseline", "audited"):
        metrics = payload[path]
        for key in (
            "layout",
            "depth",
            "size",
            "two_qubit_gates",
            "swap_gates",
            "duration_s",
            "esp",
            "esp_gate_only",
            "esp_idle_only",
            "total_idle_s",
            "worst_t2_on_layout_s",
            "worst_edge_error",
            "readout_error_sum",
        ):
            assert key in metrics, f"missing {path}.{key}"


def test_json_reports_calibration_provenance(json_result):
    payload = json.loads(json_result.stdout)
    assert payload["calibration"]["source"] == "fake_backend"
    assert payload["calibration"]["age_hours"] > 0
    assert payload["backend"]["num_qubits"] == 127
    assert any("reference snapshot" in w for w in payload["warnings"])


def test_json_settings_echo_the_seed_list(json_result):
    payload = json.loads(json_result.stdout)
    settings = payload["settings"]
    assert settings["seeds"] == 2
    assert settings["seed_list"] == [42, 43]
    assert settings["error_map"]["t_est_s"] > 0


def test_progress_is_ndjson_on_stderr(examples_dir):
    result = runner.invoke(
        app, ["run", str(examples_dir / "ghz7.py"), "--json", *FAST]
    )
    assert result.exit_code == 0
    lines = [line for line in result.stderr.splitlines() if line.strip()]
    assert lines, "no progress emitted"
    for line in lines:
        event = json.loads(line)  # every stderr line must be a JSON object
        assert "event" in event and "ts" in event
    assert any(json.loads(line)["event"] == "step" for line in lines)


def test_quiet_silences_progress(examples_dir):
    result = runner.invoke(
        app, ["run", str(examples_dir / "ghz7.py"), "--json", "--quiet", *FAST]
    )
    assert result.exit_code == 0
    assert result.stderr.strip() == ""


def test_terminal_report_renders(examples_dir):
    result = runner.invoke(app, ["run", str(examples_dir / "ghz7.py"), *FAST])
    assert result.exit_code == 0
    assert "Transpilation audit" in result.stdout
    assert "ESP (estimated success prob.)" in result.stdout
    assert "baseline layout" in result.stdout


def test_topology_map_is_drawn_by_default(examples_dir):
    result = runner.invoke(app, ["run", str(examples_dir / "ghz7.py"), "--seeds", "1"])
    assert result.exit_code == 0
    assert "Device topology" in result.stdout
    assert "T2 tiers" in result.stdout
    assert "markers:" in result.stdout


# ---------------------------------------------------------------------------
# Failure paths must still honour the output contract
# ---------------------------------------------------------------------------


def test_missing_circuit_file_json(tmp_path):
    result = runner.invoke(app, ["run", str(tmp_path / "nope.py"), "--json", *FAST])
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["kind"] == "circuit_load_error"


def test_missing_circuit_file_human(tmp_path):
    result = runner.invoke(app, ["run", str(tmp_path / "nope.py"), *FAST])
    assert result.exit_code == 2
    assert "error:" in result.stderr


def test_bad_circuit_file_json(examples_dir):
    result = runner.invoke(
        app, ["run", str(examples_dir / "bad_circuit.py"), "--json", *FAST]
    )
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "No `qc`" in payload["error"]["message"]
    assert payload["error"]["hint"]


def test_unknown_backend_json(examples_dir):
    result = runner.invoke(
        app,
        [
            "run",
            str(examples_dir / "ghz7.py"),
            "--backend",
            "fake_not_a_machine",
            "--json",
            *FAST,
        ],
    )
    assert result.exit_code == 3
    payload = json.loads(result.stdout)
    assert payload["error"]["kind"] == "calibration_error"


def test_control_flow_circuit_is_rejected_with_a_hint(tmp_path):
    path = tmp_path / "cf.py"
    path.write_text(
        "from qiskit import QuantumCircuit\n"
        "qc = QuantumCircuit(2, 1)\n"
        "qc.h(0)\n"
        "qc.measure(0, 0)\n"
        "with qc.if_test((qc.clbits[0], 1)):\n"
        "    qc.x(1)\n"
    )
    result = runner.invoke(app, ["run", str(path), "--json", *FAST])
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["error"]["kind"] == "unsupported_circuit"
    assert "--allow-control-flow" in payload["error"]["hint"]


def test_calibration_subcommand_json():
    result = runner.invoke(app, ["calibration", "fake_sherbrooke", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["num_qubits"] == 127
    assert len(payload["edges"]) == 144


def test_calibration_subcommand_human():
    result = runner.invoke(app, ["calibration", "fake_sherbrooke"])
    assert result.exit_code == 0
    assert "127 qubits" in result.stdout


def _stale_live(monkeypatch, hours: float):
    """Make resolve_calibration hand back a fake backend labelled as stale live data."""
    import datetime as _dt

    from q_audit import cli as cli_module
    from q_audit.calibration import load_fake_backend

    snapshot, backend = load_fake_backend("fake_sherbrooke")
    stale = snapshot.model_copy(
        update={
            "source": "live",
            "captured_at": _dt.datetime.now(_dt.timezone.utc)
            - _dt.timedelta(hours=hours),
        }
    )
    monkeypatch.setattr(
        cli_module, "resolve_calibration", lambda *a, **k: (stale, backend)
    )


def test_stale_live_calibration_is_refused(monkeypatch, examples_dir):
    _stale_live(monkeypatch, 30)
    result = runner.invoke(app, ["run", str(examples_dir / "ghz7.py"), "--json", *FAST])
    assert result.exit_code == 3
    payload = json.loads(result.stdout)
    assert payload["error"]["kind"] == "stale_calibration"
    assert "--allow-stale" in payload["error"]["hint"]


def test_allow_stale_proceeds_with_a_warning(monkeypatch, examples_dir):
    _stale_live(monkeypatch, 30)
    result = runner.invoke(
        app,
        ["run", str(examples_dir / "ghz7.py"), "--json", "--allow-stale", *FAST],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert any("STALE CALIBRATION ACCEPTED" in w for w in payload["warnings"])


def test_fresh_live_calibration_needs_no_flag(monkeypatch, examples_dir):
    _stale_live(monkeypatch, 0.1)
    result = runner.invoke(app, ["run", str(examples_dir / "ghz7.py"), "--json", *FAST])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["warnings"] == []


def test_json_reports_a_recommendation(examples_dir):
    result = runner.invoke(app, ["run", str(examples_dir / "ghz7.py"), "--json", *FAST])
    payload = json.loads(result.stdout)
    comparison = payload["comparison"]
    assert comparison["recommendation"] in ("adopt_audited", "keep_baseline")
    assert comparison["audited_strategy"] in ("vf2_pinned", "post_layout_relocate")
    assert comparison["strategy_esp"]


def test_qasm_stdin_end_to_end(examples_dir):
    # CliRunner installs its own stdin, so feed it through `input=`.
    result = runner.invoke(
        app, ["run", "-", "--json", *FAST], input=(examples_dir / "bell.qasm").read_text()
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["circuit"]["origin"] == "<stdin>"
    assert payload["circuit"]["num_qubits"] == 2


def test_alpha_changes_the_injected_map(examples_dir):
    payloads = []
    for alpha in ("0.0", "5.0"):
        result = runner.invoke(
            app,
            ["run", str(examples_dir / "ghz7.py"), "--json", "--alpha", alpha, *FAST],
        )
        assert result.exit_code == 0
        payloads.append(json.loads(result.stdout))
    a0, a5 = (p["settings"]["error_map"] for p in payloads)
    assert a0["alpha"] == 0.0 and a5["alpha"] == 5.0
    assert a0["diagonal_median"] != a5["diagonal_median"]


def test_map_scaling_is_echoed(examples_dir):
    result = runner.invoke(
        app,
        ["run", str(examples_dir / "ghz7.py"), "--json", "--map-scaling", "raw", *FAST],
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["settings"]["map_scaling"] == "raw"


def test_bad_map_scaling_is_a_clean_error(examples_dir):
    result = runner.invoke(
        app,
        ["run", str(examples_dir / "ghz7.py"), "--json", "--map-scaling", "nonsense", *FAST],
    )
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is False


def test_json_flags_dead_hardware_on_the_trivial_layout(examples_dir):
    """-O 1 uses TrivialLayout, which on fake_sherbrooke hits a dead coupler."""
    result = runner.invoke(
        app,
        ["run", str(examples_dir / "ghz7.py"), "--json", "-O", "1", "--seeds", "2",
         "--no-topology"],
    )
    assert result.exit_code == 0
    comparison = json.loads(result.stdout)["comparison"]
    assert comparison["baseline_unusable_instructions"], "dead coupler not reported"
    assert comparison["audited_unusable_instructions"] == []
    assert comparison["esp_baseline"] == 0.0
    # A zero baseline has no meaningful percentage; the field must say so.
    assert comparison["esp_delta_pct"] is None
    assert comparison["recommendation"] == "adopt_audited"


def test_terminal_report_shouts_about_dead_hardware(examples_dir):
    result = runner.invoke(
        app,
        ["run", str(examples_dir / "ghz7.py"), "-O", "1", "--seeds", "2", "--no-topology"],
    )
    assert result.exit_code == 0
    assert "DEAD HARDWARE" in result.stdout
    assert "from zero" in result.stdout


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--alpha", "-1"),
        ("--seeds", "0"),
        ("--schedule", "sideways"),
        ("--map-scaling", "bogus"),
    ],
)
def test_bad_flag_values_are_user_errors(examples_dir, flag, value):
    result = runner.invoke(
        app, ["run", str(examples_dir / "ghz7.py"), "--json", flag, value]
    )
    assert result.exit_code == 2, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
