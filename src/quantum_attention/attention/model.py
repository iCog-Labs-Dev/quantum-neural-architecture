
r"""The Quantum Self-Attention Neural Network (QSANN) model definition."""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml


def generate_observable(num_qubits: int, num_terms: int) -> List[List]:
    """
        Generate a list of observables for a given number of qubits.
    """
    # Each observable entry is a 2-element list: [coefficient, label]
    ob = [[1.0, f'z{idx:d}'] for idx in range(num_qubits)]
    ob.extend([[1.0, f'y{idx:d}'] for idx in range(num_qubits)])
    ob.extend([[1.0, f'x{idx:d}'] for idx in range(num_qubits)])

    if len(ob) >= num_terms:
        return ob[:num_terms]

    ob.extend(ob * (num_terms // len(ob) - 1))
    ob.extend(ob[: num_terms % len(ob)])
    return ob


def _qml_observable(op: List) -> Tuple[float, qml.operation.Operator]:
    """Defining an observable."""
    coeff, label = op
    axis = label[0]
    wire = int(label[1:])
    if axis == "x":
        return coeff, qml.PauliX(wire)
    if axis == "y":
        return coeff, qml.PauliY(wire)
    if axis == "z":
        return coeff, qml.PauliZ(wire)
    raise ValueError(f"Unknown observable label: {label}")


class QSANN(nn.Module):
    """
        Quantum Self-Attention Neural Network (QSANN).
        implementation using torch + pennylane.
    """

    def __init__(
        self,
        num_qubits: int,
        len_vocab: int,
        num_layers: int,
        depth_ebd: int,
        depth_query: int,
        depth_key: int,
        depth_value: int,
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.len_vocab = len_vocab
        self.num_layers = num_layers
        self.depth_ebd = depth_ebd
        self.depth_query = depth_query
        self.depth_key = depth_key
        self.depth_value = depth_value

        self.device = qml.device("default.qubit", wires=num_qubits)

        # Embedding parameters
        self.embedding_param = nn.Parameter(
            torch.empty(len_vocab, num_qubits * (depth_ebd * 2 + 1), 2).uniform_(-np.pi, np.pi)
        )

        # Final prediction head (same as original)
        self.weight = nn.Parameter(
            torch.randn(num_qubits * (depth_ebd * 2 + 1) * 2) * 0.001
        )
        self.bias = nn.Parameter(torch.randn(1) * 0.001)

        # Trainable circuit parameters for query/key/value circuits (per layer)
        self.query_weights = nn.Parameter(
            torch.randn(num_layers, depth_query, num_qubits, 2) * (1 / np.sqrt(num_qubits))
        )
        self.key_weights = nn.Parameter(
            torch.randn(num_layers, depth_key, num_qubits, 2) * (1 / np.sqrt(num_qubits))
        )
        self.value_weights = nn.Parameter(
            torch.randn(num_layers, depth_value, num_qubits, 2) * (1 / np.sqrt(num_qubits))
        )

        # Observables for measurements (query/key are the first observable)
        observables = generate_observable(self.num_qubits, int(self.embedding_param[0].numel()))
        self._obs_coeffs_ops = [_qml_observable(o) for o in observables]

        # This QNode returns all expectation values for the current circuit.
        @qml.qnode(self.device, interface="torch")
        def _qnode(emb_params, circ_weights):
            self._apply_embedding(emb_params)
            self._apply_circuit(circ_weights)
            # Return a list of expectation values for each observable (measurement objects)
            return [qml.expval(c * o) for c, o in self._obs_coeffs_ops]

        self._qnode = _qnode

    def _apply_embedding(self, params: torch.Tensor) -> None:
        """Apply the embedding circuit to the current quantum state."""
        # params shape: (num_qubits * (depth_ebd * 2 + 1), 2)
        idx = 0
        for d in range(self.depth_ebd):
            for q in range(self.num_qubits):
                qubits_idx = [q, (q + 1) % self.num_qubits]
                p = params[idx]
                qml.RX(p[0], wires=qubits_idx[0])
                qml.RX(p[1], wires=qubits_idx[1])
                idx += 1

                p = params[idx]
                qml.RY(p[0], wires=qubits_idx[0])
                qml.RY(p[1], wires=qubits_idx[1])
                idx += 1

                qml.CNOT(wires=qubits_idx)

        for q in range(self.num_qubits):
            p = params[idx]
            qml.RX(p[0], wires=q)
            qml.RY(p[1], wires=q)
            idx += 1

    def _apply_circuit(self, weights: torch.Tensor) -> None:
        """Apply a parameterized query/key/value circuit to the current quantum state."""
        # weights shape: (depth, num_qubits, 2)
        for l in range(weights.shape[0]):
            for q in range(self.num_qubits):
                qml.RY(weights[l, q, 0], wires=q)
                qml.RZ(weights[l, q, 1], wires=q)

            for q in range(self.num_qubits):
                qml.CNOT(wires=[q, (q + 1) % self.num_qubits])

    @staticmethod
    def _attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """Attention computation lifted from the original implementation."""
        diff = keys - query.unsqueeze(0)
        alpha = torch.exp(-diff**2)

        # Broadcast alpha over the value dimension when needed (e.g., scalar keys, vector values)
        if alpha.ndim == 1:
            alpha = alpha.unsqueeze(1)

        alpha_sum = alpha.sum(dim=0, keepdim=True)
        return (alpha * values).sum(dim=0) / (alpha_sum + eps) * np.pi

    def forward(self, batch_text: List[List[int]]) -> List[torch.Tensor]:
        predictions: List[torch.Tensor] = []

        for text in batch_text:
            text_feature = [self.embedding_param[word] for word in text]

            for layer_idx in range(self.num_layers):
                queries: List[torch.Tensor] = []
                keys: List[torch.Tensor] = []
                values: List[torch.Tensor] = []

                for emb_params in text_feature:
                    obs_query = self._qnode(emb_params, self.query_weights[layer_idx])[0]
                    obs_key = self._qnode(emb_params, self.key_weights[layer_idx])[0]
                    obs_value = self._qnode(emb_params, self.value_weights[layer_idx])

                    # Convert QNode outputs into torch tensors (avoid copy when already tensor)
                    if isinstance(obs_query, torch.Tensor):
                        queries.append(obs_query)
                    else:
                        queries.append(torch.as_tensor(obs_query, dtype=torch.get_default_dtype()))

                    if isinstance(obs_key, torch.Tensor):
                        keys.append(obs_key)
                    else:
                        keys.append(torch.as_tensor(obs_key, dtype=torch.get_default_dtype()))

                    if isinstance(obs_value, torch.Tensor):
                        values.append(obs_value)
                    elif isinstance(obs_value, (list, tuple)) and all(isinstance(v, torch.Tensor) for v in obs_value):
                        values.append(torch.stack(obs_value))
                    elif isinstance(obs_value, np.ndarray) and obs_value.dtype == object and obs_value.size > 0 and isinstance(obs_value[0], torch.Tensor):
                        values.append(torch.stack(list(obs_value)))
                    else:
                        values.append(torch.as_tensor(obs_value, dtype=torch.get_default_dtype()))

                keys = torch.stack(keys)
                values = torch.stack(values)

                feature: List[torch.Tensor] = []
                for i, query in enumerate(queries):
                    weighted = self._attention(query, keys, values)
                    weighted = weighted.reshape(self.embedding_param[0].shape)
                    feature.append(weighted)

                text_feature = feature

            output = torch.flatten(sum(text_feature) / len(text_feature))
            # Ensure consistent dtype between output and parameters
            if output.dtype != self.weight.dtype:
                output = output.to(self.weight.dtype)
            logits = output @ self.weight + self.bias
            predictions.append(torch.sigmoid(logits))

        return predictions
