"""
Classical optimisation loop for VQE.

Replaces the Trainer from vqc_fnn.  There are no datasets and no labels --
the circuit output (molecular energy in Hartrees) IS the cost function.
The optimizer drives that energy as low as physically possible.
"""

import pennylane as qml
from pennylane import numpy as np


_OPTIMIZERS = {
    "adam": qml.AdamOptimizer,
    "gd": qml.GradientDescentOptimizer,
    "nesterov": qml.NesterovMomentumOptimizer,
}


class EnergyMinimizer:
    """
    Parameters
    ----------
    model : VQEModel
        Must expose ``forward(params)`` and ``get_weight_shape()``.
    optimizer_type : str
        One of 'adam', 'gd', 'nesterov'.
    stepsize : float
        Learning rate.
    """

    def __init__(self, model, optimizer_type="gd", stepsize=0.4):
        self.model = model

        key = optimizer_type.lower()
        if key not in _OPTIMIZERS:
            raise ValueError(
                f"Unsupported optimizer: '{optimizer_type}'. "
                f"Choose from {list(_OPTIMIZERS)}"
            )
        self.opt = _OPTIMIZERS[key](stepsize=stepsize)

    # ------------------------------------------------------------------
    # Cost -- no data, no labels, just raw energy
    # ------------------------------------------------------------------

    def cost_function(self, params):
        return self.model.forward(params)

    # ------------------------------------------------------------------
    # Optimisation loop
    # ------------------------------------------------------------------

    def fit(self, epochs=100, conv_tol=1e-6, verbose_every=5):
        """
        Run the VQE optimisation.

        Parameters
        ----------
        epochs : int
            Maximum number of optimisation steps.
        conv_tol : float
            Stop early when |E_{n} - E_{n-1}| < conv_tol.
        verbose_every : int
            Print progress every *verbose_every* epochs (0 = silent).

        Returns
        -------
        dict  {params, energy_history, ground_state_energy}
        """
        shape = self.model.get_weight_shape()
        params = np.zeros(shape, requires_grad=True)

        energy_history = [float(self.cost_function(params))]

        if verbose_every:
            print(f"Step   0 | Energy = {energy_history[0]:.8f} Ha")

        for step in range(1, epochs + 1):
            params = self.opt.step(self.cost_function, params)
            energy = float(self.cost_function(params))
            energy_history.append(energy)

            if verbose_every and step % verbose_every == 0:
                print(f"Step {step:4d} | Energy = {energy:.8f} Ha")

            if abs(energy_history[-1] - energy_history[-2]) < conv_tol:
                if verbose_every:
                    print(
                        f"Converged at step {step} "
                        f"(delta < {conv_tol})"
                    )
                break

        ground_state_energy = energy_history[-1]
        if verbose_every:
            print(f"\nGround state energy = {ground_state_energy:.8f} Ha")

        return {
            "params": params,
            "energy_history": energy_history,
            "ground_state_energy": ground_state_energy,
        }
