"""
Converts a QUBO Q matrix into PennyLane Hamiltonians suitable for QAOA.

The standard substitution  x_i = (1 - Z_i) / 2  maps binary variables
to Pauli-Z eigenvalues, turning the quadratic  x^T Q x  into an Ising
Hamiltonian that a quantum circuit can evaluate directly.
"""

import numpy as np
import pennylane as qml


class ProblemFormulator:
    """
    Translates the QUBO matrix produced by ``ConceptLattice.set_cover()``
    into PennyLane operator objects consumed by QAOAAnsatz and QAOAModel.

    Parameters
    ----------
    Q : ndarray of shape (n, n)
        QUBO cost matrix.  Entry Q[i][j] encodes the quadratic penalty
        between binary decision variables x_i and x_j.
    """

    def __init__(self, Q):
        self.Q = np.array(Q, dtype=float)
        self.n_qubits = self.Q.shape[0]

        self.cost_hamiltonian = None
        self.mixer_hamiltonian = None
        self.offset = 0.0

        self._build()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_cost_hamiltonian(self):
        return self.cost_hamiltonian

    def get_mixer_hamiltonian(self):
        return self.mixer_hamiltonian

    def get_offset(self):
        """Constant energy shift dropped from the Hamiltonian."""
        return self.offset

    # ------------------------------------------------------------------
    # QUBO -> Ising conversion
    # ------------------------------------------------------------------

    def _build(self):
        """Run the substitution and store both Hamiltonians."""
        self.cost_hamiltonian, self.offset = self.qubo_to_hamiltonian(self.Q)
        self.mixer_hamiltonian = self.build_mixer(self.n_qubits)

    @staticmethod
    def qubo_to_hamiltonian(Q):
        """
        Convert a QUBO matrix to a PennyLane Hamiltonian.

        Substitution:  x_i = (1 - Z_i) / 2

        Diagonal term   Q_ii * x_i   = Q_ii * (1 - Z_i) / 2
                                      = Q_ii/2  -  Q_ii/2 * Z_i

        Off-diagonal     Q_ij * x_i * x_j  (i != j)
                        = Q_ij * (1-Z_i)/2 * (1-Z_j)/2
                        = Q_ij/4 * (1 - Z_i - Z_j + Z_i Z_j)

        Returns
        -------
        hamiltonian : qml.Hamiltonian
        offset      : float   (constant Identity contribution)
        """
        n = Q.shape[0]
        offset = 0.0
        z_coeffs = np.zeros(n)
        zz_coeffs = {}

        for i in range(n):
            for j in range(n):
                if i == j:
                    offset += Q[i, i] / 2.0
                    z_coeffs[i] -= Q[i, i] / 2.0
                else:
                    offset += Q[i, j] / 4.0
                    z_coeffs[i] -= Q[i, j] / 4.0
                    z_coeffs[j] -= Q[i, j] / 4.0

                    pair = (min(i, j), max(i, j))
                    zz_coeffs[pair] = zz_coeffs.get(pair, 0.0) + Q[i, j] / 4.0

        coeffs = []
        ops = []

        for i in range(n):
            if abs(z_coeffs[i]) > 1e-12:
                coeffs.append(z_coeffs[i])
                ops.append(qml.PauliZ(i))

        for (i, j), c in zz_coeffs.items():
            if abs(c) > 1e-12:
                coeffs.append(c)
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

        if not ops:
            coeffs.append(0.0)
            ops.append(qml.Identity(0))

        hamiltonian = qml.Hamiltonian(coeffs, ops)
        return hamiltonian, offset

    @staticmethod
    def build_mixer(n_qubits):
        """Standard X-mixer:  H_M = sum_i X_i."""
        coeffs = [1.0] * n_qubits
        ops = [qml.PauliX(i) for i in range(n_qubits)]
        return qml.Hamiltonian(coeffs, ops)
