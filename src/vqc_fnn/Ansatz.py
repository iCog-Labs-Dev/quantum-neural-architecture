import pennylane as qml


SUPPORTED_METHODS = ("basic", "strong", "random")


class AnsatzLayer:
    """
    Handles the trainable quantum gates (the variational layers).
    Supported methods: 'basic', 'strong', 'random'
    """

    def __init__(self, method="basic", n_layers=3, rotation=None):

        self.method = method.lower()
        self.n_layers = n_layers

        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported ansatz method: '{method}'. "
                f"Choose from {SUPPORTED_METHODS}"
            )

        if rotation is not None:
            self.rotation = rotation
        else:
            self.rotation = qml.RX if self.method == "basic" else None

    def apply(self, weights, wires):
        """
        Applies the chosen parameterized layers to the quantum tape."""

        if self.method == "basic":
            
            kwargs = {"weights": weights, "wires": wires}
            if self.rotation is not None:
                kwargs["rotation"] = self.rotation
            qml.BasicEntanglerLayers(**kwargs)

        elif self.method == "strong":
            qml.StronglyEntanglingLayers(weights=weights, wires=wires)

        elif self.method == "random":
            kwargs = {"weights": weights, "wires": wires}
            if self.rotation is not None:
                kwargs["rotations"] = [self.rotation] * len(wires)
            qml.RandomLayers(**kwargs)

    def get_weight_shape(self, n_qubits):

        """
        Dynamically calculates the exact tensor shape required for the weights
        to avoid a matrix mismatch error during tialization.
        """
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
