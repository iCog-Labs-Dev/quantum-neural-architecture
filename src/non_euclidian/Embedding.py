import pennylane as qml
import math
import numpy as np


SUPPORTED_METHODS = ("angle", "amplitude", "spherical", "cyclical")


class EmbeddingLayer:
    """
    Geometry-preserving encoding of classical data into quantum states.

    Supported methods
    -----------------
    angle      : Standard AngleEmbedding (1 feature per qubit via RY/RX/RZ).
    amplitude  : AmplitudeEmbedding (2^n features into n qubits).
    spherical  : Maps paired (theta, phi) coordinates onto the Bloch sphere
                 using RY(theta) then RZ(phi) per qubit.  Preserves the
                 wrap-around topology so 359 deg and 1 deg stay neighbours.
    cyclical   : Maps each scalar periodic feature to a single qubit via RZ.
                 RZ is 2-pi-periodic, so the embedding inherently wraps.
    """

    def __init__(self, method="angle", rotation="Y"):
        self.method = method.lower()
        self.rotation = rotation

        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported embedding method: '{method}'. "
                f"Choose from {SUPPORTED_METHODS}"
            )

    def apply(self, features, wires):
        """Applies the chosen embedding to the quantum tape."""

        if self.method == "angle":
            qml.AngleEmbedding(features=features, wires=wires, rotation=self.rotation)

        elif self.method == "amplitude":
            feats = self._maybe_pad(features, 2 ** len(wires))
            qml.AmplitudeEmbedding(features=feats, wires=wires, normalize=True)

        elif self.method == "spherical":
            self._apply_spherical(features, wires)

        elif self.method == "cyclical":
            self._apply_cyclical(features, wires)

    
    # Spherical embedding
    @staticmethod
    def _apply_spherical(features, wires):
        """
        Expects *features* to contain pairs: [theta0, phi0, theta1, phi1, ...].
        Each consecutive (theta, phi) pair is mapped to one qubit via RY then RZ,
        placing the data point on the corresponding location of the Bloch sphere.
        """
        n_pairs = len(features) // 2
        for i in range(n_pairs):
            theta = features[2 * i]
            phi = features[2 * i + 1]
            qml.RY(theta, wires=wires[i])
            qml.RZ(phi, wires=wires[i])

    
    # Cyclical embedding
    @staticmethod
    def _apply_cyclical(features, wires):
        """
        Each feature is already an angle in [0, 2*pi].
        RZ is 2-pi-periodic so the wrap-around is automatic.
        """
        for i, wire in enumerate(wires):
            qml.RZ(features[i], wires=wire)

   
    # Metadatas
    def get_required_qubits(self, num_features):
        """Calculate how many qubits are needed for the given feature count."""
        if self.method == "angle":
            return num_features
        elif self.method == "amplitude":
            return math.ceil(math.log2(max(num_features, 2)))
        elif self.method == "spherical":
            return num_features // 2
        elif self.method == "cyclical":
            return num_features

    @staticmethod
    def _maybe_pad(features, target_length):
        """Pad a feature vector with zeros to reach *target_length*."""
        if hasattr(features, "__len__") and len(features) >= target_length:
            return features
        pad_width = target_length - len(features)
        return np.pad(features, (0, pad_width), mode="constant")
