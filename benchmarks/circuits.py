"""the benchmark circuits.

picked to spread over the two things that matter for a layout audit: how dense
the interaction graph is (does vf2 find an exact embedding or do we fall back
to sabre + post layout) and how long the circuit runs (how much idle time is
there to save).

one thing worth knowing about the fidelity numbers: QFT on |0...0> gives a
uniform output distribution, and noise also gives you something roughly
uniform, so hellinger fidelity barely moves however badly you compile it.
those cases are here for their compiler metrics only and are flagged
sharp=False. the mirror circuits (U, barrier, U-dagger) have an exact delta
function reference so they're the ones that can actually see a difference
"""

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate, efficient_su2


@dataclass(frozen=True)
class BenchCircuit:
    name: str
    build: object
    sharp: bool  # can hellinger fidelity actually tell anything apart here
    note: str = ""


def _bind_all(qc, seed):
    if not qc.parameters:
        return qc
    rng = np.random.default_rng(seed)
    values = rng.uniform(0, 2 * np.pi, len(qc.parameters))
    return qc.assign_parameters(dict(zip(qc.parameters, values)))


def ghz(n):
    qc = QuantumCircuit(n, name=f"ghz{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def qft(n):
    qc = QuantumCircuit(n, name=f"qft{n}")
    qc.append(QFTGate(n), range(n))
    qc.measure_all()
    return qc


def qft_mirror(n):
    # qft then its inverse. the barrier stops the optimiser collapsing the pair
    # into nothing. ideal output is exactly |0...0>
    qc = QuantumCircuit(n, name=f"qft_mirror{n}")
    qc.append(QFTGate(n), range(n))
    qc.barrier()
    qc.append(QFTGate(n).inverse(), range(n))
    qc.measure_all()
    return qc


def su2(n, reps, seed):
    qc = _bind_all(efficient_su2(n, reps=reps, entanglement="linear"), seed)
    qc = qc.copy()
    qc.name = f"su2_{n}x{reps}"
    qc.measure_all()
    return qc


def qaoa_3regular(n, seed, layers=1):
    """one layer of qaoa on a random 3 regular graph.

    sparse but not a line, which is the interesting middle case
    """
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


def random_circuit(n, depth, seed):
    from qiskit.circuit.random import random_circuit as _rc

    qc = _rc(n, depth, max_operands=2, measure=False, seed=seed)
    qc = qc.copy()
    qc.name = f"random{n}d{depth}"
    qc.measure_all()
    return qc


def random_mirror(n, depth, seed):
    from qiskit.circuit.random import random_circuit as _rc

    body = _rc(n, depth, max_operands=2, measure=False, seed=seed)
    qc = QuantumCircuit(n, name=f"random_mirror{n}d{depth}")
    qc.compose(body, inplace=True)
    qc.barrier()
    qc.compose(body.inverse(), inplace=True)
    qc.measure_all()
    return qc


SUITE = {
    c.name: c
    for c in [
        BenchCircuit("ghz7", lambda: ghz(7), sharp=True,
                     note="line graph, vf2 embeds it exactly"),
        BenchCircuit("ghz10", lambda: ghz(10), sharp=True,
                     note="longer line, more idle time piled on qubit 0"),
        BenchCircuit("qft7", lambda: qft(7), sharp=False,
                     note="K7 graph, forces the sabre fallback"),
        BenchCircuit("qft10", lambda: qft(10), sharp=False,
                     note="K10, heaviest routing case in here"),
        BenchCircuit("qft_mirror7", lambda: qft_mirror(7), sharp=True,
                     note="dense AND measurable, delta function reference"),
        BenchCircuit("su2_6x3", lambda: su2(6, 3, seed=7), sharp=True,
                     note="EfficientSU2, linear entanglement, params bound"),
        BenchCircuit("qaoa6", lambda: qaoa_3regular(6, seed=3), sharp=True,
                     note="3 regular graph, one qaoa layer"),
        BenchCircuit("random7d20", lambda: random_circuit(7, 20, seed=1234), sharp=True,
                     note="fixed seed random circuit"),
        BenchCircuit("random_mirror7d10", lambda: random_mirror(7, 10, seed=99), sharp=True,
                     note="random mirror, delta function reference"),
    ]
}

# qft10 and random_mirror7d10 are in SUITE but not run by default, they're slow
DEFAULT_SUITE = ["ghz7", "ghz10", "qft7", "qft_mirror7", "su2_6x3", "qaoa6", "random7d20"]
