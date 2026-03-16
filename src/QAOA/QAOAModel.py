"""
Central QAOA model.

Replaces VQCModel from vqc_fnn.  Stitches StatePreparation + QAOAAnsatz
together into PennyLane QNodes and exposes two execution modes:

* ``forward(params)``  -- returns the expectation value of the cost
  Hamiltonian (used during optimisation).
* ``sample(params, shots)`` -- returns computational-basis samples so the
  most probable bitstring can be read out as the solution.
"""

import pennylane as qml
from pennylane import numpy as np

from StatePreparation import StatePreparation
from QAOAAnsatz import QAOAAnsatz


class QAOAModel:
    """
    Parameters
    ----------
    n_qubits : int
        One qubit per binary decision variable (one per concept).
    qaoa_ansatz : QAOAAnsatz
        Pre-configured ansatz carrying cost and mixer Hamiltonians.
    cost_hamiltonian : qml.Hamiltonian
        Same cost Hamiltonian used inside the ansatz (needed for measurement).
    state_prep : StatePreparation or None
        Defaults to the standard Hadamard preparation.
    device_name : str
        PennyLane backend.
    """

    def __init__(
        self,
        n_qubits,
        qaoa_ansatz,
        cost_hamiltonian,
        state_prep=None,
        device_name="default.qubit",
    ):
        self.n_qubits = n_qubits
        self.ansatz = qaoa_ansatz
        self.cost_h = cost_hamiltonian
        self.state_prep = state_prep or StatePreparation()

        self.device = qml.device(device_name, wires=n_qubits)

        self._expval_qnode = qml.QNode(self._expval_circuit, self.device)

    # ------------------------------------------------------------------
    # Circuit definitions
    # ------------------------------------------------------------------

    def _expval_circuit(self, params):
        """Circuit that returns <H_cost>."""
        wires = range(self.n_qubits)
        self.state_prep.apply(wires)
        gammas, betas = params[0], params[1]
        self.ansatz.apply(gammas, betas)
        return qml.expval(self.cost_h)

    def _sample_circuit(self, params):
        """Circuit that returns computational-basis samples."""
        wires = range(self.n_qubits)
        self.state_prep.apply(wires)
        gammas, betas = params[0], params[1]
        self.ansatz.apply(gammas, betas)
        return qml.sample()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, params):
        """Evaluate the cost Hamiltonian expectation (used by the optimiser)."""
        return self._expval_qnode(params)

    def sample(self, params, shots=1024):
        """
        Run the optimised circuit and return raw bitstring samples.

        Returns
        -------
        ndarray of shape (shots, n_qubits)
            Each row is a measured bitstring (0s and 1s).
        """
        sample_dev = qml.device("default.qubit", wires=self.n_qubits, shots=shots)
        sample_qnode = qml.QNode(self._sample_circuit, sample_dev)
        return sample_qnode(params)
