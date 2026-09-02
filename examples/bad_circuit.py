# broken on purpose, there's no `qc` here. used to test the error path

from qiskit import QuantumCircuit

not_the_circuit = QuantumCircuit(2)
not_the_circuit.h(0)
