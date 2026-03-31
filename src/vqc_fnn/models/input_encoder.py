
import pennylane as qml
import numpy as np
from typing import Callable, Literal

ArrayLike = np.ndarray


class InputEncoder:
    """
    Stateless quantum data encoder.

    Responsibilities:
    - Define how classical data is embedded into a quantum circuit
    - Apply embedding gates inside a QNode
    - NO device creation
    - NO QNode creation
    - NO state return
    """

    def __init__(
        self,
        n_qubits: int,
        embedding_type: Literal["angle", "amplitude"] = "angle",
        rotation: Literal["X", "Y", "Z"] = "Y",
    ):
        self.n_qubits = n_qubits
        self.embedding_type = embedding_type
        self.rotation = rotation

        if embedding_type == "amplitude":
            max_dim = 2 ** n_qubits
            self.max_amplitude_dim = max_dim

    def __call__(self, x: ArrayLike):
        """
        Apply embedding operations for a single data sample.

        This function is meant to be called INSIDE a QNode.
        """
        x = np.asarray(x, dtype=np.float64).flatten()

        if self.embedding_type == "angle":
            if len(x) != self.n_qubits:
                raise ValueError(
                    f"Angle embedding requires input dimension {self.n_qubits}, got {len(x)}"
                )

            qml.AngleEmbedding(
                x,
                wires=range(self.n_qubits),
                rotation=self.rotation
            )

        elif self.embedding_type == "amplitude":
            if len(x) > self.max_amplitude_dim:
                raise ValueError(
                    f"Amplitude embedding requires len(x) <= {self.max_amplitude_dim}"
                )

            qml.AmplitudeEmbedding(
                x,
                wires=range(self.n_qubits),
                normalize=True,
                pad_with=0.0
            )

        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
            
            
            
