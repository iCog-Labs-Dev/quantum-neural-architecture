from pennylane import numpy as pnp
import pennylane as qml
import numpy as np
from .Ansatz import AnsatzLayer


class BornMachineModel:
    """
    Quantum Born Machine.

    forward(params) returns the full 2^n probability vector (the Born distribution).
    sample(params, shots) returns raw bitstring samples.
    """

    def __init__(self, n_qubits, ansatz=None, device_name="default.qubit"):
        self.n_qubits = n_qubits
        self.wires = list(range(n_qubits))
        self.ansatz = ansatz if ansatz is not None else AnsatzLayer()

        self.weight_shape = self.ansatz.get_weight_shape(n_qubits)

        self.dev = qml.device(device_name, wires=n_qubits)
        self._build_prob_circuit()

    def _build_prob_circuit(self):
        """Build the QNode that returns the full probability vector."""
        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            self.ansatz.apply(params, wires=self.wires)
            return qml.probs(wires=self.wires)

        self._prob_circuit = circuit

    def forward(self, params):
        """
        Evaluate the circuit and return the Born distribution.

        Returns
        -------
        ndarray of shape (2^n_qubits,)
        """
        return self._prob_circuit(params)

    def sample(self, params, shots=1024):
        """
        Draw independent bitstring samples from the Born distribution.

        Returns
        -------
        ndarray of shape (shots, n_qubits)
        """
        dev = qml.device("default.qubit", wires=self.n_qubits, shots=shots)

        @qml.qnode(dev, interface="autograd")
        def sample_circuit(params):
            self.ansatz.apply(params, wires=self.wires)
            return qml.sample(wires=self.wires)

        raw = sample_circuit(params)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        return raw

    def init_params(self, seed=None):
        """Initialise random parameters for the circuit."""
        rng = np.random.default_rng(seed)
        raw = rng.uniform(0,2 * np.pi, size = self.weight_shape)
        return pnp.array(raw, dtype = pnp.float64, requires_grad = True)

    def param_count(self):
        """Total number of scalar trainable parameters."""
        return int(np.prod(self.weight_shape))
