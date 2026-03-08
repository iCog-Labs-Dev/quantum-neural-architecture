import pennylane as qml
import torch
from torch import nn
from typing import List

from .projections import QuantumProjection
from ..utils.encodings import controlled_amplitude_encoding

class BlockEncoding(nn.Module):
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

        self.W_Q = QuantumProjection(n_data_qubits, n_layers)
        self.W_K = QuantumProjection(n_data_qubits, n_layers)
        self.W_V = QuantumProjection(n_data_qubits, n_layers)

       
        self.total_wires = (2 * n_index_qubits) + (3 * n_data_qubits)
        self.device = qml.device(device_type, wires=self.total_wires)

    def _apply_UA(self, x_vectors: List[torch.Tensor]):
        
      
        idx_i = list(range(self.n_index_qubits))
        data_q = list(range(self.n_index_qubits, self.n_index_qubits + self.n_data_qubits))
        
        idx_j = list(range(self.n_index_qubits + self.n_data_qubits, 2 * self.n_index_qubits + self.n_data_qubits))
        data_k = list(range(2 * self.n_index_qubits + self.n_data_qubits, 2 * self.n_index_qubits + 2 * self.n_data_qubits))
        
        data_v = list(range(2 * self.n_index_qubits + 2 * self.n_data_qubits, self.total_wires))

       
        for q in idx_i + idx_j:
            qml.Hadamard(wires=q)
        for idx, x in enumerate(x_vectors):
            bin_idx = [int(b) for b in format(idx, f"0{self.n_index_qubits}b")]
           
            controlled_amplitude_encoding(x, data_q, idx_i, bin_idx)
            controlled_amplitude_encoding(x, data_k, idx_j, bin_idx)
            controlled_amplitude_encoding(x, data_v, idx_j, bin_idx)

        self.W_Q.apply(wires=data_q)
        self.W_K.apply(wires=data_k)
        self.W_V.apply(wires=data_v)