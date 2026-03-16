import pennylane as qml
from pennylane import numpy as np

from .Ansatz import AnsatzLayer
from .Embedding import EmbeddingLayer


SUPPORTED_MEASUREMENTS = ("expval", "probs", "expval_all")


class VQCModel:
    """
    Central variational quantum circuit model.

    Assembles embedding, ansatz, and measurement into a single PennyLane QNode
    and provides a ``forward()`` interface consumed by the Trainer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    embedding : EmbeddingLayer
        Configured embedding strategy.
    ansatz : AnsatzLayer
        Configured ansatz strategy.
    measurement : str
        One of 'expval' (PauliZ on qubit 0), 'probs' (probability vector),
        or 'expval_all' (PauliZ on every qubit).
    device_name : str
        PennyLane device backend.
    dropout_rate : float
        Probability of zeroing each rotation weight during training (0 = off).
    """

    def __init__(
        self,
        n_qubits,
        embedding=None,
        ansatz=None,
        measurement="expval",
        device_name="default.qubit",
        dropout_rate=0.0,
    ):
        self.n_qubits = n_qubits
        self.embedding = embedding or EmbeddingLayer()
        self.ansatz = ansatz or AnsatzLayer()
        self.dropout_rate = dropout_rate
        self.training = True

        if measurement not in SUPPORTED_MEASUREMENTS:
            raise ValueError(
                f"Unsupported measurement: '{measurement}'. "
                f"Choose from {SUPPORTED_MEASUREMENTS}"
            )
        self.measurement = measurement

        self.device = qml.device(device_name, wires=n_qubits)
        self._qnode = qml.QNode(self._circuit, self.device)

    # ------------------------------------------------------------------
    # Quantum dropout
    # ------------------------------------------------------------------

    def _apply_dropout(self, weights):
        """Zero-out rotation angles with probability ``dropout_rate``."""
        if self.training and self.dropout_rate > 0:
            mask = np.random.binomial(
                1, 1 - self.dropout_rate, size=weights.shape
            )
            return weights * mask
        return weights

    # ------------------------------------------------------------------
    # Circuit definition
    # ------------------------------------------------------------------

    def _circuit(self, features, weights):
        wires = range(self.n_qubits)
        self.embedding.apply(features, wires)
        self.ansatz.apply(weights, wires)
        return self._measure(wires)

    def _measure(self, wires):
        if self.measurement == "expval":
            return qml.expval(qml.PauliZ(wires[0]))
        elif self.measurement == "probs":
            return qml.probs(wires=wires)
        elif self.measurement == "expval_all":
            return [qml.expval(qml.PauliZ(w)) for w in wires]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x, weights):
        """Run one input sample through the circuit and return the result."""
        w = self._apply_dropout(weights)
        return self._qnode(x, w)

    def train(self):
        """Enable training mode (dropout active)."""
        self.training = True

    def eval(self):
        """Enable evaluation mode (dropout disabled)."""
        self.training = False
