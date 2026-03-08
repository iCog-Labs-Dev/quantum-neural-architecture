import pennylane as qml
import torch
from typing import Callable, Optional


class QuantumProjection:
    """
    Parametrized quantum projection used for Query, Key, or Value.

    Args:
        n_qubits (int): number of qubits in the data register
        n_layers (int): number of variational layers
        ansatz_fn (Callable): optional custom ansatz
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        ansatz_fn: Optional[Callable] = None,
    ):

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        if ansatz_fn is None:
            self.ansatz_fn = self.default_ansatz
            weight_shape = qml.BasicEntanglerLayers.shape(
                n_layers=n_layers,
                n_wires=n_qubits
            )
        else:
            self.ansatz_fn = ansatz_fn
            weight_shape = (n_layers, n_qubits)

       
        self.weights = torch.nn.Parameter(
            torch.randn(*weight_shape) * torch.pi
        )


    def default_ansatz(self, weights, wires):
        """
        Default PQC for projection using BasicEntanglerLayers.
        """
        qml.BasicEntanglerLayers(weights, wires=wires)



    def apply(self, wires):
        """
        Apply the projection circuit.

        Args:
            wires (list[int]): qubits representing the input vector
        """
        self.ansatz_fn(self.weights, wires=wires)