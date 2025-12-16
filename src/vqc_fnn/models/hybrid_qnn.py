import torch
from torch import nn
import numpy as np
from typing import Literal, Optional, Callable

from .input_encoder import InputEncoder
from .vqc_layer import VQCLayer


class HybridQNN(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network.

    Architecture:
    - Classical input → Quantum embedding (InputEncoder)
    - Variational Quantum Circuit (VQCLayer)
    - Classical linear output layer

    Args:
        num_features (int): Number of classical input features.
        num_classes (int): Number of output classes.
        n_vqc_layers (int): Number of layers in the quantum ansatz.
        embedding_type (Literal['angle', 'amplitude']): Type of quantum embedding.
        gate_type (Literal['X', 'Y', 'Z']): Rotation gate type for angle embedding.
        encoding_strategy (Literal['single', 'reupload']): Single encoding or data re-uploading.
        device_type (str): PennyLane device type (default.qubit, lightning.qubit, etc.).
        ansatz_fn (Optional[Callable]): Custom ansatz function for VQC layer.
        measurement_fn (Optional[Callable]): Custom measurement function.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        n_vqc_layers: int = 2,
        embedding_type: Literal["angle", "amplitude"] = "angle",
        gate_type: Literal["X", "Y", "Z"] = "Y",
        encoding_strategy: Literal["single", "reupload"] = "reupload",
        device_type: str = "default.qubit",
        ansatz_fn: Optional[Callable] = None,
        measurement_fn: Optional[Callable] = None,
    ):
        super().__init__()

        self.embedding_type = embedding_type
        self.gate_type = gate_type
        # Determine number of qubits
       
        if embedding_type == "amplitude":
            # Number of wires for amplitude embedding
            n_qubits = int(np.ceil(np.log2(num_features)))
        else:
            # Angle embedding: one qubit per feature
            n_qubits = num_features

        # Initialize InputEncoder
      
        self.encoder = InputEncoder(
            n_qubits=n_qubits,
            embedding_type=embedding_type,
            rotation=gate_type
        )

 
        # Initialize VQC layer
       
        self.quantum_layer = VQCLayer(
            n_qubits=n_qubits,
            encoder=self.encoder,
            n_layers=n_vqc_layers,
            encoding_strategy=encoding_strategy,
            ansatz_fn=ansatz_fn,
            measurement_fn=measurement_fn,
            device_type=device_type
        )

     
        # Classical linear head
       
        # Output dimension of quantum layer = n_qubits (Z expectation per qubit)
        self.linear_head = nn.Linear(n_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor (batch_size, num_features)

        Returns:
            torch.Tensor: Logits (batch_size, num_classes)
        """
        # Pass through quantum layer
        q_out = self.quantum_layer(x)

        # Pass through classical head
        return self.linear_head(q_out)
