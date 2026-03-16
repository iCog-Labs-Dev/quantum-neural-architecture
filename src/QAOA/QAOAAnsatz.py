"""
QAOA variational ansatz.

Replaces the generic AnsatzLayer from vqc_fnn.  Instead of
StronglyEntanglingLayers / BasicEntanglerLayers, it alternates between
a *cost layer* (driven by the problem Hamiltonian) and a *mixer layer*
for ``p`` rounds.  The trainable parameters are the gamma and beta angles.
"""

import pennylane as qml


class QAOAAnsatz:
    """
    Parameters
    ----------
    cost_hamiltonian : qml.Hamiltonian
        Ising cost operator built from the QUBO matrix.
    mixer_hamiltonian : qml.Hamiltonian
        Mixer operator (default: sum of Pauli-X).
    p : int
        Number of QAOA layers (circuit depth).
    """

    def __init__(self, cost_hamiltonian, mixer_hamiltonian, p=1):
        self.cost_h = cost_hamiltonian
        self.mixer_h = mixer_hamiltonian
        self.p = p

    def apply(self, gammas, betas):
        """
        Apply p alternating cost / mixer layers.

        Parameters
        ----------
        gammas : array of shape (p,)
            Cost-layer angles.
        betas : array of shape (p,)
            Mixer-layer angles.
        """
        for i in range(self.p):
            qml.qaoa.cost_layer(gammas[i], self.cost_h)
            qml.qaoa.mixer_layer(betas[i], self.mixer_h)

    def get_weight_shape(self):
        """Returns ``(2, p)`` -- row 0 for gammas, row 1 for betas."""
        return (2, self.p)
