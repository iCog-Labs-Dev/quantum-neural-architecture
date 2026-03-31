import torch
from torch import nn
import pennylane as qml
import numpy as np
from typing import Callable, Optional, Literal

from .input_encoder import InputEncoder


class VQCLayer(nn.Module):
    """
    Variational Quantum Circuit layer.

    Responsibilities:
    - Own device and QNode
    - Apply encoding (single or re-uploading)
    - Apply ansatz
    - Measure observables or return full quantum state (optional)
    - Return classical outputs by default
    """

    def __init__(
        self,
        n_qubits: int,
        encoder: InputEncoder,
        n_layers: int = 1,
        encoding_strategy: Literal["single", "reupload"] = "reupload",
        ansatz_fn: Optional[Callable] = None,
        measurement_fn: Optional[Callable[[int], list]] = None,
        device_type: str = "default.qubit",
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding_strategy = encoding_strategy
        self.encoder = encoder

        self.device = qml.device(device_type, wires=n_qubits)

        # Default ansatz: BasicEntanglerLayers
        if ansatz_fn is None:
            self.ansatz_fn = lambda weights: qml.BasicEntanglerLayers(
                weights, wires=range(n_qubits)
            )
            weight_shape = qml.BasicEntanglerLayers.shape(
                n_layers=n_layers, n_wires=n_qubits
            )
        else:
            self.ansatz_fn = ansatz_fn
            weight_shape = (n_layers, n_qubits)

        # Trainable quantum parameters
        self.weights = nn.Parameter(torch.randn(*weight_shape) * np.pi)

        # Default measurement: Z expectation on all qubits
        if measurement_fn is None:
            self.measurement_fn = lambda: [
                qml.expval(qml.PauliZ(i)) for i in range(n_qubits)
            ]
        else:
            self.measurement_fn = measurement_fn

        # Define QNode ONCE
        self.qnode = qml.QNode(
            self._circuit,
            self.device,
            interface="torch",
            diff_method="parameter-shift",
        )

    # --------------------------------------------------
    # Quantum circuit definition
    # --------------------------------------------------
    def _circuit(self, x, weights, return_state=False):
        """
        x: 1D tensor (n_features)
        weights: trainable parameters, shape (n_layers, n_qubits)
        return_state: bool, if True return full quantum state
        """
        if self.encoding_strategy == "single":
            self.encoder(x)
            self.ansatz_fn(weights)
        elif self.encoding_strategy == "reupload":
            for layer in range(self.n_layers):
                self.encoder(x)
                self.ansatz_fn(weights[layer].unsqueeze(0))
        else:
            raise ValueError(f"Unknown encoding strategy: {self.encoding_strategy}")

        if return_state:
            return qml.state()
        else:
            return self.measurement_fn()

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------
    def forward(self, x: torch.Tensor, return_state: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): (batch_size, n_features) or (n_features,)
            return_state (bool): if True, returns full quantum states instead of observables

        Returns:
            torch.Tensor: classical outputs (batch_size, n_measurements) 
                          or quantum states (batch_size, 2**n_qubits)
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)

        outputs = []
        for sample in x:
            out = self.qnode(sample, self.weights, return_state=return_state)

            # If measurement returns list/tuple, stack it
            if isinstance(out, (list, tuple)):
                out = torch.stack(out)

            outputs.append(out)

        return torch.stack(outputs).to(x.dtype)
