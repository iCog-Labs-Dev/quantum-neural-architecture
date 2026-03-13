"""
State preparation for QAOA.

Replaces the data-driven EmbeddingLayer from vqc_fnn.  Instead of encoding
classical features, it puts all qubits into a uniform superposition so the
circuit begins by exploring every candidate solution simultaneously.
"""

import pennylane as qml


class StatePreparation:
    """Apply a Hadamard gate to every qubit (uniform superposition over all 2^n basis states)."""

    @staticmethod
    def apply(wires):
        for wire in wires:
            qml.Hadamard(wires=wire)
