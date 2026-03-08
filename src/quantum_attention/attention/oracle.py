# src/attention/oracle.py
import pennylane as qml
import torch
import numpy as np
from typing import List, Callable, Optional
from ..utils.encodings import controlled_amplitude_encoding


class QuantumAttentionOracle:
    """
    Quantum Attention Oracle using a swap test to compute <Q_i|K_j>.

    Args:
        n_data_qubits (int): Number of qubits per vector
        n_index_qubits (int): Number of qubits to encode sequence indices
        U_Q (Callable): Parametrized PQC for query projection
        U_K (Callable): Parametrized PQC for key projection
        device_type (str): PennyLane device type
    """

    def __init__(
        self,
        n_data_qubits: int,
        n_index_qubits: int,
        U_Q: Optional[Callable] = None,
        U_K: Optional[Callable] = None,
        device_type: str = "default.qubit",
    ):
        self.n_data_qubits = n_data_qubits
        self.n_index_qubits = n_index_qubits
        self.U_Q = U_Q if U_Q is not None else self.default_pqc
        self.U_K = U_K if U_K is not None else self.default_pqc

        self.n_qubits = 2 * n_data_qubits + 2 * n_index_qubits + 1

        self.device = qml.device(device_type, wires=self.n_qubits)

        self.qnode = qml.QNode(self._circuit, self.device, interface="torch", diff_method="parameter-shift")


    def default_pqc(self, weights, wires):
        qml.BasicEntanglerLayers(weights, wires=wires)


    def _circuit(
        self,
        x_vectors: List[np.ndarray],
        weights_Q: torch.Tensor,
        weights_K: torch.Tensor,
    ):
        """
        Quantum circuit for attention oracle.

        Returns:
            torch.Tensor: attention scores ⟨Q_i|K_j⟩
        """
        n_vectors = len(x_vectors)
        n_data = self.n_data_qubits
        n_index = self.n_index_qubits

        if n_vectors > 2 ** n_index:
            raise ValueError("Too many vectors for the given index qubits.")

      
        index_wires_i = list(range(n_index))
        data_wires_i = list(range(n_index, n_index + n_data))

        index_wires_j = list(range(n_index + n_data, 2 * n_index + n_data))
        data_wires_j = list(range(2 * n_index + n_data, 2 * n_index + 2 * n_data))

        ancilla = [self.n_qubits - 1]

       
        for q in index_wires_i + index_wires_j:
            qml.Hadamard(wires=q)

        for idx, x in enumerate(x_vectors):
            binary_idx = [int(b) for b in format(idx, f"0{n_index}b")]

            controlled_amplitude_encoding(x, data_wires_i, index_wires_i, binary_idx)
            controlled_amplitude_encoding(x, data_wires_j, index_wires_j, binary_idx)

      
        self.U_Q(weights_Q, wires=data_wires_i)
        self.U_K(weights_K, wires=data_wires_j)

       
        qml.Hadamard(wires=ancilla[0])

        for q1, q2 in zip(data_wires_i, data_wires_j):
            qml.CSWAP(wires=[ancilla[0], q1, q2])

        qml.Hadamard(wires=ancilla[0])

        return qml.expval(qml.PauliZ(ancilla[0]))

    def forward(
        self,
        x_vectors: List[np.ndarray],
        weights_Q: torch.Tensor,
        weights_K: torch.Tensor,
    ):
        """
        Execute oracle and return attention scores.
        """
        return self.qnode(x_vectors, weights_Q, weights_K)