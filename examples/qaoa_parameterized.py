"""qaoa-ish circuit with unbound parameters.

shows what q-audit does with free parameters. layout and routing don't care
about the angles but scheduling and esp do, so they get bound to fixed
pseudo-random values (seeded off circuit + parameter name) and the report says
it did that. re-running gives the same numbers
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
