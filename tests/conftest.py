"""Shared fixtures. Backend construction is session-scoped: instantiating
FakeSherbrooke parses a 127-qubit calibration file and is far too slow to
repeat per test."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=DeprecationWarning)

EXAMPLES = ROOT / "examples"


@pytest.fixture(scope="session")
def fake_backend():
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

    return FakeSherbrooke()


@pytest.fixture(scope="session")
def snapshot(fake_backend):
    from q_audit.calibration import snapshot_from_target

    return snapshot_from_target(
        fake_backend.target, backend_name="fake_sherbrooke", source="fake_backend"
    )


@pytest.fixture(scope="session")
def target(fake_backend):
    return fake_backend.target


@pytest.fixture
def ghz7():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(7, name="ghz7")
    qc.h(0)
    for i in range(6):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


@pytest.fixture(scope="session")
def examples_dir():
    return EXAMPLES
