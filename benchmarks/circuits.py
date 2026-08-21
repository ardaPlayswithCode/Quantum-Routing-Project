"""The benchmark circuit suite.

Chosen for spread across the two axes that matter to a layout audit:
interaction-graph density (does VF2 find an exact embedding, or do we fall back
to SABRE + post-layout?) and circuit duration (how much idle time is there to
save?).

A note on which of these can be *measured* with Hellinger fidelity: QFT applied
to |0...0> produces a uniform computational-basis distribution, and noise also
produces something near-uniform, so its Hellinger fidelity barely moves no
matter how badly the circuit is compiled.  Those cases are kept for their
compiler metrics (depth, SWAPs, ESP, duration) and flagged ``sharp=False``.
The mirror circuits (U, barrier, U-dagger) have an exact delta-function
reference and are the sensitive fidelity probes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate, efficient_su2


@dataclass(frozen=True)
class BenchCircuit:
    name: str
    build: Callable[[], QuantumCircuit]
    sharp: bool  # is Hellinger fidelity a meaningful discriminator here?
    note: str = ""


def _bind_all(qc: QuantumCircuit, seed: int) -> QuantumCircuit:
    if not qc.parameters:
        return qc
    rng = np.random.default_rng(seed)
    values = rng.uniform(0, 2 * np.pi, len(qc.parameters))
    return qc.assign_parameters(dict(zip(qc.parameters, values)))


def ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"ghz{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def qft(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"qft{n}")
    qc.append(QFTGate(n), range(n))
    qc.measure_all()
    return qc


def qft_mirror(n: int) -> QuantumCircuit:
    """QFT then its inverse, separated by a barrier so the optimiser cannot
    collapse the pair.  Ideal output is |0...0> exactly."""
    qc = QuantumCircuit(n, name=f"qft_mirror{n}")
    qc.append(QFTGate(n), range(n))
    qc.barrier()
    qc.append(QFTGate(n).inverse(), range(n))
    qc.measure_all()
    return qc


def su2(n: int, reps: int, seed: int) -> QuantumCircuit:
    qc = _bind_all(efficient_su2(n, reps=reps, entanglement="linear"), seed)
    qc = qc.copy()
    qc.name = f"su2_{n}x{reps}"
    qc.measure_all()
    return qc


def qaoa_3regular(n: int, seed: int, layers: int = 1) -> QuantumCircuit:
    """One-layer QAOA on a random 3-regular graph -- the canonical
    'sparse but not a line' interaction graph."""
    import rustworkx as rx

    graph = rx.undirected_gnm_random_graph(n, 3 * n // 2, seed=seed)
    edges = list(graph.edge_list())
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n, name=f"qaoa{n}")
    qc.h(range(n))
    for _ in range(layers):
        gamma = float(rng.uniform(0, np.pi))
        beta = float(rng.uniform(0, np.pi))
        for a, b in edges:
            qc.cx(a, b)
            qc.rz(2 * gamma, b)
            qc.cx(a, b)
        for q in range(n):
            qc.rx(2 * beta, q)
    qc.measure_all()
    return qc


def random_circuit(n: int, depth: int, seed: int) -> QuantumCircuit:
    from qiskit.circuit.random import random_circuit as _rc

    qc = _rc(n, depth, max_operands=2, measure=False, seed=seed)
    qc = qc.copy()
    qc.name = f"random{n}d{depth}"
    qc.measure_all()
    return qc


def random_mirror(n: int, depth: int, seed: int) -> QuantumCircuit:
    from qiskit.circuit.random import random_circuit as _rc

    body = _rc(n, depth, max_operands=2, measure=False, seed=seed)
    qc = QuantumCircuit(n, name=f"random_mirror{n}d{depth}")
    qc.compose(body, inplace=True)
    qc.barrier()
    qc.compose(body.inverse(), inplace=True)
    qc.measure_all()
    return qc


SUITE: dict[str, BenchCircuit] = {
    c.name: c
    for c in [
        BenchCircuit("ghz7", lambda: ghz(7), sharp=True,
                     note="line interaction graph; VF2 embeds exactly"),
        BenchCircuit("ghz10", lambda: ghz(10), sharp=True,
                     note="longer line, more idle time on qubit 0"),
        BenchCircuit("qft7", lambda: qft(7), sharp=False,
                     note="K7 interaction graph; forces the SABRE fallback"),
        BenchCircuit("qft10", lambda: qft(10), sharp=False,
                     note="K10; heaviest routing case in the suite"),
        BenchCircuit("qft_mirror7", lambda: qft_mirror(7), sharp=True,
                     note="dense AND measurable: delta-function reference"),
        BenchCircuit("su2_6x3", lambda: su2(6, 3, seed=7), sharp=True,
                     note="EfficientSU2, linear entanglement, params bound"),
        BenchCircuit("qaoa6", lambda: qaoa_3regular(6, seed=3), sharp=True,
                     note="3-regular graph, one QAOA layer"),
        BenchCircuit("random7d20", lambda: random_circuit(7, 20, seed=1234), sharp=True,
                     note="fixed-seed random circuit"),
        BenchCircuit("random_mirror7d10", lambda: random_mirror(7, 10, seed=99), sharp=True,
                     note="random mirror; delta-function reference"),
    ]
}

DEFAULT_SUITE = ["ghz7", "ghz10", "qft7", "qft_mirror7", "su2_6x3", "qaoa6", "random7d20"]
