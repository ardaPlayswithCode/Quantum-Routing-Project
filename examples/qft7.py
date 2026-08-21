"""7-qubit QFT followed by a full measurement -- the reference demo circuit."""

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate

qc = QuantumCircuit(7, name="qft7")
qc.h(range(7))
qc.append(QFTGate(7), range(7))
qc.measure_all()
