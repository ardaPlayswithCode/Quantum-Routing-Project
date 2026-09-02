# 7 qubit ghz. interaction graph is a line so vf2 finds a perfect embedding on
# heavy-hex, which means the layout choice is purely about noise

from qiskit import QuantumCircuit

qc = QuantumCircuit(7, name="ghz7")
qc.h(0)
for i in range(6):
    qc.cx(i, i + 1)
qc.measure_all()
