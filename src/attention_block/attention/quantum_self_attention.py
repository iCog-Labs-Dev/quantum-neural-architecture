import torch
import torch.nn as nn
import torch.nn.functional as F
from ..quantum.quantum_kernel import QuantumKernel


class QuantumSelfAttention(nn.Module):

    def __init__(
        self,
        n_qubits,
        n_layers,
        device,
        embed_dim,
        max_seq_len=128
    ):

        super().__init__()

        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.max_seq_len = max_seq_len

        self.input_proj = nn.Linear(
            embed_dim,
            n_qubits
        )

        self.value_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.q_kernel = QuantumKernel(
            n_qubits,
            n_layers,
            device
        )

        self.out_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.theta = nn.Parameter(
            torch.randn(n_layers, n_qubits, 2) * 0.1
        )

        self.phi = nn.Parameter(
            torch.randn(max_seq_len) * 0.1
        )


    def forward(self, X):
        batch_size, seq_len, _ = X.shape
        assert seq_len <= self.max_seq_len

        quantum_input = self.input_proj(X)
        V = self.value_proj(X)

        outputs = []
        attn_maps = []

        for b in range(batch_size):
            K = self.q_kernel.compute_kernel_matrix(
                quantum_input[b],
                self.theta
            )

            K = K / torch.sqrt(
                torch.tensor(self.n_qubits, dtype=K.dtype)
            )

            phi_slice = self.phi[:seq_len]
            phi_diff = phi_slice.unsqueeze(0) - phi_slice.unsqueeze(1)
            K = K + torch.cos(phi_diff)

            attn = F.softmax(K, dim=-1)

            
            attn = attn.to(V[b].dtype)  

            out = torch.matmul(attn, V[b])

            outputs.append(out)
            attn_maps.append(attn)

        outputs = torch.stack(outputs)
        attn_maps = torch.stack(attn_maps)

        return self.out_proj(outputs), attn_maps