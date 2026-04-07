import numpy as np
import pennylane as qml


class ClassicalBaseline:
    """
    Classical benchmark for VQE: computes

    - Hartree-Fock energy
    - Exact (Full Configuration Interaction / FCI) energy
    - Correlation energy (FCI - HF)
    """

    def __init__(self, environment):
        self.env = environment
        self.n_qubits = environment.n_qubits

    def get_hf_state_vector(self):
        """Convert Hartree-Fock occupation list to full 2^N vector."""
        binary_str = "".join(map(str, self.env.hf_state))
        state_idx = int(binary_str, 2)

        vector = np.zeros(2**self.n_qubits, dtype=complex)
        vector[state_idx] = 1.0
        return vector

    def calculate(self):
        """Compute HF, FCI energies and correlation energy."""

        h_matrix = qml.matrix(
            self.env.hamiltonian,
            wire_order=range(self.n_qubits)
        )

        eigenvalues = np.linalg.eigvalsh(h_matrix)
        fci_energy = float(np.min(eigenvalues))

        hf_vec = self.get_hf_state_vector()
        hf_energy = float(np.real(np.vdot(hf_vec, h_matrix @ hf_vec)))

        return {
            "hf_energy": hf_energy,
            "fci_energy": fci_energy,
            "correlation_energy": fci_energy - hf_energy,
        }