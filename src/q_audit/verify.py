"""aer simulation harness. dev only, never imported on the optimizer path.

qiskit-aer is an optional dependency so every import of it happens inside a
function, that way `q-audit run` still works on a machine without aer.

the padding matters more than you'd think. aer applies noise per instruction,
so a transpiled circuit with no explicit delays simulates as if idle qubits
were perfectly preserved, which is exactly the effect this whole tool is about.
so we schedule and PadDelay everything before simulating. on ghz7 that moves
fidelity from 0.888 (unpadded) to 0.838 (padded). without it the benchmark is
blind to decoherence
"""

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
        import qiskit_aer
    except ImportError as exc:
        raise AerUnavailableError(
            "qiskit-aer isn't installed, --verify and the benchmarks need it",
            hint="uv pip install 'qiskit-aer==0.17.2'",
        ) from exc
    return qiskit_aer


def ideal_distribution(circuit):
    """exact noise free output of the logical circuit.

    statevector rather than a noiseless shot run so the reference has no
    sampling error of its own
    """
    from qiskit.quantum_info import Statevector

    if circuit.num_qubits > MAX_STATEVECTOR_QUBITS:
        raise AerUnavailableError(
            f"ideal statevector needs <= {MAX_STATEVECTOR_QUBITS} qubits, "
            f"this circuit has {circuit.num_qubits}"
        )
    stripped = circuit.remove_final_measurements(inplace=False)
    probs = Statevector(stripped).probabilities_dict()
    return {str(k): float(v) for k, v in probs.items() if v > 1e-12}


def pad_for_simulation(circuit, target, *, schedule_method="asap"):
    """turn idle time into actual delay instructions so aer can see it"""
    analysis = ASAPScheduleAnalysis if schedule_method == "asap" else ALAPScheduleAnalysis
    pm = PassManager(
        [
            TimeUnitConversion(target=target),
            analysis(target=target),
            PadDelay(target=target),
        ]
    )
    return pm.run(circuit)


def noisy_counts(circuit, backend, *, shots=8192, seed_simulator=1234, pad=True,
                 schedule_method="asap", noise_model=None):
    require_aer()
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel

    if noise_model is None:
        noise_model = NoiseModel.from_backend(backend)
    if pad:
        to_run = pad_for_simulation(circuit, backend.target, schedule_method=schedule_method)
    else:
        to_run = circuit
    sim = AerSimulator(noise_model=noise_model)
    result = sim.run(to_run, shots=shots, seed_simulator=seed_simulator).result()
    return {str(k): int(v) for k, v in result.get_counts().items()}


def simulated_fidelity(transpiled, ideal, backend, *, shots=8192, seed_simulator=1234,
                       pad=True, schedule_method="asap", noise_model=None):
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


def ensure_measured(circuit):
    """aer needs counts, so add a measurement if there isn't one"""
    if any(inst.operation.name == "measure" for inst in circuit.data):
        return circuit
    copy = circuit.copy()
    copy.measure_all()
    return copy
