"""7-qubit GHZ state. Its interaction graph is a line, so VF2 finds a perfect
subgraph embedding on heavy-hex -- the layout choice is pure noise-awareness."""

from qiskit import QuantumCircuit

qc = QuantumCircuit(7, name="ghz7")
qc.h(0)
for i in range(6):
    qc.cx(i, i + 1)
qc.measure_all()
