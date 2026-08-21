"""Deliberately broken: defines no `qc`. Used to exercise the error path."""

from qiskit import QuantumCircuit

not_the_circuit = QuantumCircuit(2)
not_the_circuit.h(0)
