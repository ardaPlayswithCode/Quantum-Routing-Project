"""Get a ``QuantumCircuit`` from a file, a path, or stdin.

``QuantumCircuit`` is the pipeline currency end to end.  We deliberately do not
route through QIR: every pass we need (layout, routing, scheduling) is defined
over Qiskit's DAG, and a QIR round trip would only add a lossy hop.

Supported inputs
----------------
* ``*.py``     -- executed; must leave a ``QuantumCircuit`` in ``qc``
                  (``circuit`` and a zero-arg ``build_circuit``/``get_circuit``
                  are accepted as fallbacks)
* ``*.qasm``   -- OpenQASM; the version is sniffed from the header
* ``*.qasm3``  -- OpenQASM 3
* ``-``        -- read QASM text from stdin

NOTE: loading a ``.py`` circuit executes that file.  Only point q-audit at
circuit files you would run yourself.
"""

from __future__ import annotations

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


def sniff_qasm_version(text: str) -> int:
    """Return 2 or 3 based on the OPENQASM header; default to 3 when absent."""
    head = "\n".join(
        line for line in text.splitlines()[:40] if not line.strip().startswith("//")
    )
    if _QASM2_HEADER.search(head):
        return 2
    if _QASM3_HEADER.search(head):
        return 3
    # No header at all: OpenQASM 3 is the current standard, and its parser gives
    # much better diagnostics on malformed input.
    return 3


def load_qasm_text(text: str, *, origin: str = "<stdin>") -> QuantumCircuit:
    version = sniff_qasm_version(text)
    if version == 2:
        from qiskit import qasm2

        try:
            return qasm2.loads(text)
        except Exception as exc:  # noqa: BLE001 - parser raises many types
            raise CircuitLoadError(
                f"Failed to parse {origin} as OpenQASM 2: {exc}"
            ) from exc
    try:
        from qiskit import qasm3
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise CircuitLoadError("OpenQASM 3 support is not installed.") from exc
    try:
        return qasm3.loads(text)
    except Exception as exc:  # noqa: BLE001
        raise CircuitLoadError(
            f"Failed to parse {origin} as OpenQASM 3: {exc}",
            hint="If this is OpenQASM 2, add an 'OPENQASM 2.0;' header.",
        ) from exc


def load_python_module(path: Path) -> QuantumCircuit:
    sys_path_added = str(path.parent.resolve())
    inserted = sys_path_added not in sys.path
    if inserted:
        sys.path.insert(0, sys_path_added)
    try:
        namespace = runpy.run_path(str(path), run_name="__q_audit_circuit__")
    except Exception as exc:  # noqa: BLE001 - user code can raise anything
        raise CircuitLoadError(
            f"Error while executing {path}: {type(exc).__name__}: {exc}"
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
            except TypeError:
                continue
            except Exception as exc:  # noqa: BLE001
                raise CircuitLoadError(f"{path}:{name}() raised {exc}") from exc
            if isinstance(obj, QuantumCircuit):
                return obj

    found = sorted(k for k, v in namespace.items() if isinstance(v, QuantumCircuit))
    hint = (
        f"Found QuantumCircuit(s) named {found}; rename one to 'qc'."
        if found
        else "Define a module-level `qc = QuantumCircuit(...)`."
    )
    raise CircuitLoadError(f"No `qc` QuantumCircuit found in {path}.", hint=hint)


def load_circuit(source: str) -> tuple[QuantumCircuit, str]:
    """Load a circuit from ``source``. Returns ``(circuit, origin_description)``."""
    if source == "-":
        text = sys.stdin.read()
        if not text.strip():
            raise CircuitLoadError("Nothing on stdin.", hint="Pipe QASM in, or pass a file path.")
        return load_qasm_text(text, origin="<stdin>"), "<stdin>"

    path = Path(source).expanduser()
    if not path.exists():
        raise CircuitLoadError(f"No such file: {source}")
    if path.is_dir():
        raise CircuitLoadError(f"{source} is a directory.")

    suffix = path.suffix.lower()
    if suffix == ".py":
        return load_python_module(path), str(path)
    if suffix in (".qasm", ".qasm2", ".qasm3", ".txt", ""):
        return load_qasm_text(path.read_text(), origin=str(path)), str(path)
    raise CircuitLoadError(
        f"Unsupported circuit file type {suffix!r}.",
        hint="Use .py (defines `qc`), .qasm, or '-' for QASM on stdin.",
    )
