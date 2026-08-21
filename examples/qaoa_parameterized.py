"""A parameterized QAOA-style circuit with *unbound* parameters.

Demonstrates q-audit's parameter handling: layout and routing do not depend on
rotation angles, but scheduling and ESP do, so the free parameters are bound to
fixed pseudo-random values (seeded by circuit + parameter name) and the report
says so. Re-running gives identical numbers.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4), (2, 5)]

gamma = Parameter("gamma")
beta = Parameter("beta")

qc = QuantumCircuit(6, name="qaoa6_param")
qc.h(range(6))
for a, b in EDGES:
    qc.cx(a, b)
    qc.rz(2 * gamma, b)
    qc.cx(a, b)
for q in range(6):
    qc.rx(2 * beta, q)
qc.measure_all()
