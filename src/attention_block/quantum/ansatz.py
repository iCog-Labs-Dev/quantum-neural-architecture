import pennylane as qml
import torch


class StronglyEntanglingAnsatz:
    """
    Data re-uploading strongly entangling ansatz.

    Encodes classical input  directly into rotations,
    combined with trainable variational parameters.

    Args:
        n_qubits (int): Number of qubits in the circuit.
        n_layers (int, optional): Number of repeated rotation + entanglement layers. Default: 2

    Attributes:
        n_params (int): Total number of trainable parameters for RX/RY rotations.
    """

    def __init__(self, n_qubits: int, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = n_layers * n_qubits * 2

    def __call__(self, x: torch.Tensor, weights: torch.Tensor):
        """
        Apply the ansatz to the input data and trainable parameters.

        Args:
            x (torch.Tensor):
                Input classical vector to encode.
                Shape: (n_qubits,)
                

            weights (torch.Tensor):
                Trainable circuit parameters.
                Shape: (n_layers, n_qubits, 2)
                weights[layer, qubit, 0] -> scaling for RX(x)
                weights[layer, qubit, 1] -> RY rotation

        """

        if x.shape[0] != self.n_qubits:
            raise ValueError(
                f"Input length {x.shape[0]} does not match n_qubits {self.n_qubits}"
            )

        for layer in range(self.n_layers):
            
            for qubit in range(self.n_qubits):
                rx_theta = weights[layer, qubit, 0] * x[qubit] 
                ry_theta = weights[layer, qubit, 1]             

                qml.RX(rx_theta, wires=qubit)
                qml.RY(ry_theta, wires=qubit)
                qml.RZ(x[qubit] ** 2, wires=qubit)  

    
            for qubit in range(self.n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])