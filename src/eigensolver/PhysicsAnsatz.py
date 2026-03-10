"""
Chemistry-aware variational ansatz for VQE.

Replaces the generic AnsatzLayer from vqc_fnn.  Instead of
StronglyEntanglingLayers (which ignore electron-number conservation),
this applies physics-motivated excitation gates.
"""

import numpy as np
import pennylane as qml


SUPPORTED_METHODS = ("double_excitation", "all_singles_doubles")


class PhysicsAnsatz:
    """
    Parameters
    ----------
    method : str
        'double_excitation' -- minimal single-parameter ansatz sufficient
            for H2 in a minimal basis (DoubleExcitation on wires 0-3).
        'all_singles_doubles' -- UCCSD-inspired template that automatically
            enumerates all single and double excitations from the
            Hartree-Fock state.  Works for larger molecules.
    n_electrons : int
        Number of electrons in the molecule.
    n_qubits : int
        Number of qubits (spin-orbitals).
    hf_state : array
        Hartree-Fock occupation vector, e.g. [1,1,0,0].
    """

    def __init__(self, method="double_excitation",
                 n_electrons=2, n_qubits=4, hf_state=None):
        self.method = method.lower()
        self.n_electrons = n_electrons
        self.n_qubits = n_qubits
        self.hf_state = hf_state

        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported ansatz method: '{method}'. "
                f"Choose from {SUPPORTED_METHODS}"
            )

        self.singles = None
        self.doubles = None
        if self.method == "all_singles_doubles":
            self.singles, self.doubles = qml.qchem.excitations(
                self.n_electrons, self.n_qubits
            )

    def apply(self, params, wires):
        """Apply the chosen ansatz gates to the quantum tape."""
        if self.method == "double_excitation":
            qml.DoubleExcitation(params[0], wires=list(wires[:4]))

        elif self.method == "all_singles_doubles":
            qml.AllSinglesDoubles(
                params, wires, self.hf_state,
                self.singles, self.doubles,
            )

    def get_weight_shape(self):
        """Return the shape of the parameter array required by this ansatz."""
        if self.method == "double_excitation":
            return (1,)
        elif self.method == "all_singles_doubles":
            n_params = len(self.singles) + len(self.doubles)
            return (n_params,)
