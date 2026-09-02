"""get a QuantumCircuit out of a file, a path or stdin.

QuantumCircuit is the currency everywhere in this project. i looked at going
through QIR but every pass we need (layout, routing, scheduling) is defined
over qiskit's dag anyway, so it would just be a lossy round trip.

what we accept:
  .py file that leaves a QuantumCircuit in `qc` (circuit / build_circuit /
  get_circuit also work), .qasm (version sniffed from the header), .qasm3,
  or "-" to read qasm off stdin.

heads up: loading a .py circuit *runs* that file. only point this at circuit
files you'd run yourself
"""

import re
import runpy
import sys
from pathlib import Path

from qiskit import QuantumCircuit

from .errors import CircuitLoadError

_QASM3_HEADER = re.compile(r"OPENQASM\s+3(\.\d+)?\s*;", re.IGNORECASE)
_QASM2_HEADER = re.compile(r"OPENQASM\s+2(\.\d+)?\s*;", re.IGNORECASE)
_CIRCUIT_NAMES = ("qc", "circuit", "CIRCUIT")
_BUILDER_NAMES = ("build_circuit", "get_circuit", "build", "main")


def sniff_qasm_version(text) -> int:
    """2 or 3 from the OPENQASM header, defaults to 3 if there isn't one"""
    head = "\n".join(
        line for line in text.splitlines()[:40] if not line.strip().startswith("//")
    )
    if _QASM2_HEADER.search(head):
        return 2
    if _QASM3_HEADER.search(head):
        return 3
    # no header. qasm3 is the current standard and its parser complains much
    # more usefully on broken input
    return 3


def load_qasm_text(text, *, origin="<stdin>"):
    version = sniff_qasm_version(text)
    if version == 2:
        from qiskit import qasm2

        try:
            return qasm2.loads(text)
        except Exception as exc:  # the parser raises about five different types
            raise CircuitLoadError(f"couldn't parse {origin} as OpenQASM 2: {exc}") from exc
    try:
        from qiskit import qasm3
    except ImportError as exc:
        raise CircuitLoadError("OpenQASM 3 support isn't installed") from exc
    try:
        return qasm3.loads(text)
    except Exception as exc:
        raise CircuitLoadError(
            f"couldn't parse {origin} as OpenQASM 3: {exc}",
            hint="if this is actually OpenQASM 2 then add an 'OPENQASM 2.0;' header",
        ) from exc


def load_python_module(path):
    # put the file's own directory on sys.path first so it can import its
    # helpers, then take it back off again
    sys_path_added = str(path.parent.resolve())
    inserted = sys_path_added not in sys.path
    if inserted:
        sys.path.insert(0, sys_path_added)
    try:
        namespace = runpy.run_path(str(path), run_name="__q_audit_circuit__")
    except Exception as exc:  # user code, could raise literally anything
        raise CircuitLoadError(
            f"error while running {path}: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if inserted:
            try:
                sys.path.remove(sys_path_added)
            except ValueError:
                pass

    for name in _CIRCUIT_NAMES:
        obj = namespace.get(name)
        if isinstance(obj, QuantumCircuit):
            return obj
    for name in _BUILDER_NAMES:
        fn = namespace.get(name)
        if callable(fn):
            try:
                obj = fn()
            except TypeError:  # takes arguments, not the builder we wanted
                continue
            except Exception as exc:
                raise CircuitLoadError(f"{path}:{name}() raised {exc}") from exc
            if isinstance(obj, QuantumCircuit):
                return obj

    found = sorted(k for k, v in namespace.items() if isinstance(v, QuantumCircuit))
    if found:
        hint = f"found QuantumCircuit(s) named {found}, rename one of them to 'qc'"
    else:
        hint = "define a module level `qc = QuantumCircuit(...)`"
    raise CircuitLoadError(f"No `qc` QuantumCircuit in {path}", hint=hint)


def load_circuit(source):
    """load a circuit from source, returns (circuit, where it came from)"""
    if source == "-":
        text = sys.stdin.read()
        if not text.strip():
            raise CircuitLoadError("nothing on stdin", hint="pipe qasm in, or pass a file path")
        return load_qasm_text(text, origin="<stdin>"), "<stdin>"

    path = Path(source).expanduser()
    if not path.exists():
        raise CircuitLoadError(f"no such file: {source}")
    if path.is_dir():
        raise CircuitLoadError(f"{source} is a directory")

    suffix = path.suffix.lower()
    if suffix == ".py":
        return load_python_module(path), str(path)
    if suffix in (".qasm", ".qasm2", ".qasm3", ".txt", ""):
        return load_qasm_text(path.read_text(), origin=str(path)), str(path)
    raise CircuitLoadError(
        f"Unsupported circuit file type {suffix!r}",
        hint="use .py (defines `qc`), .qasm, or '-' for qasm on stdin",
    )
