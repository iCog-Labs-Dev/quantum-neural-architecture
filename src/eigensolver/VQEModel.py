
import pennylane as qml
from pennylane import numpy as np

from .ChemistryEnvironment import ChemistryEnvironment
from .PhysicsAnsatz import PhysicsAnsatz


class VQEModel:
    """
    Assembles ChemistryEnvironment + PhysicsAnsatz into a PennyLane QNode
    whose output is the molecular energy in Hartrees.

    Parameters
    ----------
    environment : ChemistryEnvironment
        Provides Hamiltonian, HF state, and qubit count.
    ansatz : PhysicsAnsatz
        Provides the variational gate structure.
    device_name : str
        PennyLane backend (constrained to 'default.qubit').
    """

    def __init__(self, environment, ansatz, device_name="default.qubit"):
        self.env = environment
        self.ansatz = ansatz
        self.n_qubits = environment.n_qubits

        self.device = qml.device(device_name, wires=self.n_qubits)
        self._qnode = qml.QNode(self._circuit, self.device)

   
    # Circuit
    def _circuit(self, params):
        wires = range(self.n_qubits)
        self.env.prepare_state(wires)
        self.ansatz.apply(params, wires)
        return qml.expval(self.env.hamiltonian)

    
    # Public API
    def forward(self, params):
        """Return the molecular energy (Hartrees) for the given parameters."""
        return self._qnode(params)

    def get_weight_shape(self):
        """Delegate to the ansatz for parameter shape."""
        return self.ansatz.get_weight_shape()
