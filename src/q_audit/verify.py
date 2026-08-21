"""Aer validation harness. DEV-ONLY -- never imported on the optimizer path.

``qiskit-aer`` is an optional dependency; every import of it lives inside a
function body here so that ``q-audit run`` works on a machine without Aer.

Why padding matters
-------------------
Aer applies noise per *instruction*.  A transpiled circuit with no explicit
``delay`` instructions therefore simulates as if idle qubits were perfectly
preserved -- exactly the effect this tool exists to measure.  So every circuit
is scheduled and padded with ``PadDelay`` before simulation.  Empirically this
moves GHZ-7 on fake_sherbrooke from F=0.888 (unpadded) to F=0.838 (padded);
skipping it would make the benchmark blind to decoherence.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    ALAPScheduleAnalysis,
    ASAPScheduleAnalysis,
    PadDelay,
    TimeUnitConversion,
)

from .errors import QAuditError
from .physics import hellinger_fidelity

MAX_STATEVECTOR_QUBITS = 20


class AerUnavailableError(QAuditError):
    exit_code = 2
    kind = "aer_unavailable"


def require_aer():
    try:
        import qiskit_aer  # noqa: F401
    except ImportError as exc:
        raise AerUnavailableError(
            "qiskit-aer is not installed; --verify and the benchmarks need it.",
            hint="uv pip install 'qiskit-aer==0.17.2'",
        ) from exc
    return qiskit_aer


def ideal_distribution(circuit: QuantumCircuit) -> dict[str, float]:
    """Exact output distribution of the *logical* circuit, noise-free.

    Uses a statevector rather than a noiseless shot simulation so the reference
    carries no sampling error of its own.
    """
    from qiskit.quantum_info import Statevector

    if circuit.num_qubits > MAX_STATEVECTOR_QUBITS:
        raise AerUnavailableError(
            f"Ideal statevector needs <= {MAX_STATEVECTOR_QUBITS} qubits; "
            f"circuit has {circuit.num_qubits}."
        )
    stripped = circuit.remove_final_measurements(inplace=False)
    probs = Statevector(stripped).probabilities_dict()
    return {str(k): float(v) for k, v in probs.items() if v > 1e-12}


def pad_for_simulation(
    circuit: QuantumCircuit, target, *, schedule_method: str = "asap"
) -> QuantumCircuit:
    """Materialise idle time as explicit ``delay`` instructions."""
    analysis = ASAPScheduleAnalysis if schedule_method == "asap" else ALAPScheduleAnalysis
    pm = PassManager(
        [
            TimeUnitConversion(target=target),
            analysis(target=target),
            PadDelay(target=target),
        ]
    )
    return pm.run(circuit)


def noisy_counts(
    circuit: QuantumCircuit,
    backend,
    *,
    shots: int = 8192,
    seed_simulator: int = 1234,
    pad: bool = True,
    schedule_method: str = "asap",
    noise_model=None,
) -> dict[str, int]:
    require_aer()
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel

    if noise_model is None:
        noise_model = NoiseModel.from_backend(backend)
    to_run = (
        pad_for_simulation(circuit, backend.target, schedule_method=schedule_method)
        if pad
        else circuit
    )
    sim = AerSimulator(noise_model=noise_model)
    result = sim.run(to_run, shots=shots, seed_simulator=seed_simulator).result()
    return {str(k): int(v) for k, v in result.get_counts().items()}


def simulated_fidelity(
    transpiled: QuantumCircuit,
    ideal: dict[str, float],
    backend,
    *,
    shots: int = 8192,
    seed_simulator: int = 1234,
    pad: bool = True,
    schedule_method: str = "asap",
    noise_model=None,
) -> tuple[float, dict[str, int]]:
    counts = noisy_counts(
        transpiled,
        backend,
        shots=shots,
        seed_simulator=seed_simulator,
        pad=pad,
        schedule_method=schedule_method,
        noise_model=noise_model,
    )
    return hellinger_fidelity(ideal, {k: float(v) for k, v in counts.items()}), counts


def ensure_measured(circuit: QuantumCircuit) -> QuantumCircuit:
    """Add a full measurement if the circuit has none -- Aer needs counts."""
    if any(inst.operation.name == "measure" for inst in circuit.data):
        return circuit
    copy = circuit.copy()
    copy.measure_all()
    return copy
