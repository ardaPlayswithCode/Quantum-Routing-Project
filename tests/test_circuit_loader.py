"""Circuit ingestion: .py, OpenQASM 2/3, stdin, and the failure modes."""

from __future__ import annotations

import io
import sys

import pytest
from qiskit import QuantumCircuit

from q_audit.circuit_loader import (
    load_circuit,
    load_python_module,
    load_qasm_text,
    sniff_qasm_version,
)
from q_audit.errors import CircuitLoadError

QASM2 = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
"""

QASM3 = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c[0] = measure q[0];
c[1] = measure q[1];
"""


def test_sniff_qasm2():
    assert sniff_qasm_version(QASM2) == 2


def test_sniff_qasm3():
    assert sniff_qasm_version(QASM3) == 3


def test_sniff_defaults_to_three_when_header_absent():
    assert sniff_qasm_version("qubit[2] q;") == 3


def test_sniff_ignores_leading_comments():
    assert sniff_qasm_version("// a note\n// another\nOPENQASM 2.0;\n") == 2


def test_load_qasm2():
    qc = load_qasm_text(QASM2)
    assert qc.num_qubits == 2
    assert qc.count_ops()["cx"] == 1


def test_load_qasm3():
    qc = load_qasm_text(QASM3)
    assert qc.num_qubits == 2
    assert qc.count_ops()["cx"] == 1


def test_bad_qasm_raises_user_error():
    with pytest.raises(CircuitLoadError) as exc:
        load_qasm_text("OPENQASM 2.0;\nthis is not qasm;\n")
    assert exc.value.exit_code == 2


def test_load_python_file(tmp_path):
    path = tmp_path / "circ.py"
    path.write_text(
        "from qiskit import QuantumCircuit\nqc = QuantumCircuit(3)\nqc.h(0)\n"
    )
    qc = load_python_module(path)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 3


def test_load_python_file_accepts_circuit_alias(tmp_path):
    path = tmp_path / "circ.py"
    path.write_text("from qiskit import QuantumCircuit\ncircuit = QuantumCircuit(2)\n")
    assert load_python_module(path).num_qubits == 2


def test_load_python_file_accepts_builder_function(tmp_path):
    path = tmp_path / "circ.py"
    path.write_text(
        "from qiskit import QuantumCircuit\n"
        "def build_circuit():\n"
        "    c = QuantumCircuit(4)\n"
        "    c.h(0)\n"
        "    return c\n"
    )
    assert load_python_module(path).num_qubits == 4


def test_missing_qc_names_what_it_found(tmp_path):
    path = tmp_path / "circ.py"
    path.write_text(
        "from qiskit import QuantumCircuit\nsomething_else = QuantumCircuit(2)\n"
    )
    with pytest.raises(CircuitLoadError) as exc:
        load_python_module(path)
    assert "something_else" in (exc.value.hint or "")


def test_python_file_that_raises_is_reported_cleanly(tmp_path):
    path = tmp_path / "boom.py"
    path.write_text("raise ValueError('kaboom')\n")
    with pytest.raises(CircuitLoadError) as exc:
        load_python_module(path)
    assert "kaboom" in exc.value.message


def test_load_circuit_from_examples(examples_dir):
    qc, origin = load_circuit(str(examples_dir / "qft7.py"))
    assert qc.num_qubits == 7
    assert "qft7.py" in origin


def test_load_circuit_from_stdin(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(QASM2))
    qc, origin = load_circuit("-")
    assert qc.num_qubits == 2
    assert origin == "<stdin>"


def test_empty_stdin_is_a_user_error(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("   \n"))
    with pytest.raises(CircuitLoadError):
        load_circuit("-")


def test_missing_file():
    with pytest.raises(CircuitLoadError) as exc:
        load_circuit("/definitely/not/here.qasm")
    assert exc.value.exit_code == 2


def test_unsupported_extension(tmp_path):
    path = tmp_path / "circ.bin"
    path.write_bytes(b"\x00\x01")
    with pytest.raises(CircuitLoadError) as exc:
        load_circuit(str(path))
    assert "Unsupported" in exc.value.message
