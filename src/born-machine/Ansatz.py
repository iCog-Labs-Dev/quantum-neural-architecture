import pennylane as qml


SUPPORTED_METHODS = ("basic", "strong", "random")


class AnsatzLayer:
    """
    Trainable variational layers for the Born Machine.

    Default is 'strong' (StronglyEntanglingLayers) which provides the
    entanglement needed to access the full Hilbert space.

    Supported methods: 'basic', 'strong', 'random'
    """

    def __init__(self, method="strong", n_layers=6):
        self.method = method.lower()
        self.n_layers = n_layers

        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported ansatz method: '{method}'. "
                f"Choose from {SUPPORTED_METHODS}"
            )

    def apply(self, weights, wires):
        """Applies the chosen parameterised layers to the quantum tape."""
        if self.method == "basic":
            qml.BasicEntanglerLayers(weights=weights, wires=wires)
        elif self.method == "strong":
            qml.StronglyEntanglingLayers(weights=weights, wires=wires)
        elif self.method == "random":
            qml.RandomLayers(weights=weights, wires=wires)

    def get_weight_shape(self, n_qubits):
        """Dynamically calculates the exact tensor shape required for the weights."""
        if self.method == "basic":
            return qml.BasicEntanglerLayers.shape(
                n_layers=self.n_layers, n_wires=n_qubits
            )
        elif self.method == "strong":
            return qml.StronglyEntanglingLayers.shape(
                n_layers=self.n_layers, n_wires=n_qubits
            )
        elif self.method == "random":
            return qml.RandomLayers.shape(
                n_layers=self.n_layers, n_rotations=n_qubits
            )

    def param_count(self, n_qubits):
        """Total number of scalar trainable parameters."""
        import numpy as np
        return int(np.prod(self.get_weight_shape(n_qubits)))
