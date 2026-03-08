import torch
from torch import nn
import pennylane as qml
import numpy as np
from typing import List

from .block_encoding import BlockEncoding


class QuantumAttentionBlock(nn.Module):
    """
    Quantum Attention Block.

    Steps:
        1. Prepare |i>|j>|q_i>|k_j>|v_j> using BlockEncoding
        2. Compute similarity using Swap Test
        3. Use swap test result to control V register
        4. Measure final V register

    """

    def __init__(
        self,
        n_data_qubits: int,
        n_index_qubits: int,
        n_layers: int = 2,
        device_type: str = "default.qubit",
    ):
        super().__init__()

        self.n_data_qubits = n_data_qubits
        self.n_index_qubits = n_index_qubits

      
        self.block_encoding = BlockEncoding(
            n_data_qubits=n_data_qubits,
            n_index_qubits=n_index_qubits,
            n_layers=n_layers,
        )

        self.n_qubits = (
            2 * n_index_qubits
            + 3 * n_data_qubits
            + 1  
        )

        self.device = qml.device(device_type, wires=self.n_qubits)

        self.qnode = qml.QNode(
            self._circuit,
            self.device,
            interface="torch",
            diff_method="parameter-shift",
        )


    def _circuit(self, x_vectors: List[np.ndarray]):

        n_index = self.n_index_qubits
        n_data = self.n_data_qubits

      
        index_i = list(range(n_index))
        data_q = list(range(n_index, n_index + n_data))

        index_j = list(range(n_index + n_data, 2 * n_index + n_data))
        data_k = list(range(2 * n_index + n_data, 2 * n_index + 2 * n_data))

        data_v = list(
            range(2 * n_index + 2 * n_data, 2 * n_index + 3 * n_data)
        )

        ancilla = 2 * n_index + 3 * n_data

        

        self.block_encoding.apply(x_vectors)

      

        qml.Hadamard(wires=ancilla)

        for q, k in zip(data_q, data_k):
            qml.CSWAP(wires=[ancilla, q, k])

        qml.Hadamard(wires=ancilla)

  

        for v in data_v:
            qml.CRZ(np.pi / 2, wires=[ancilla, v])

        

        return [qml.expval(qml.PauliZ(w)) for w in data_v]

   

    def forward(self, x_vectors):

        if isinstance(x_vectors, torch.Tensor):
            x_vectors = x_vectors.detach().cpu().numpy()

        return torch.tensor(self.qnode(x_vectors))