"""
Classical optimisation loop for QAOA.

Replaces the Trainer from vqc_fnn.  Key differences:
* No labels -- the circuit output *is* the cost.
* After optimisation, a ``solve()`` method samples the circuit and
  returns the most probable bitstring as the combinatorial answer.
"""

import pennylane as qml
from pennylane import numpy as np
from collections import Counter


_OPTIMIZERS = {
    "adam": qml.AdamOptimizer,
    "gd": qml.GradientDescentOptimizer,
    "nesterov": qml.NesterovMomentumOptimizer,
    "spsa": qml.SPSAOptimizer,
}


class QAOAOptimizer:
    """
    Parameters
    ----------
    model : QAOAModel
        The assembled QAOA model.
    optimizer_type : str
        One of 'adam', 'gd', 'nesterov', 'spsa'.
    stepsize : float
        Learning rate.
    """

    def __init__(self, model, optimizer_type="adam", stepsize=0.1):
        self.model = model

        key = optimizer_type.lower()
        if key not in _OPTIMIZERS:
            raise ValueError(
                f"Unsupported optimizer: '{optimizer_type}'. "
                f"Choose from {list(_OPTIMIZERS)}"
            )
        self.opt = _OPTIMIZERS[key](stepsize=stepsize)

        self.optimal_params = None
        self.cost_history = []

    # ------------------------------------------------------------------
    # Cost (no labels -- circuit expectation IS the cost)
    # ------------------------------------------------------------------

    def cost_function(self, params):
        return self.model.forward(params)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(self, p=1, epochs=80, verbose_every=10):
        """
        Initialise gamma/beta angles and run the optimisation loop.

        Parameters
        ----------
        p : int
            QAOA depth (must match the ansatz).
        epochs : int
            Number of optimisation steps.
        verbose_every : int
            Print progress every *verbose_every* epochs (0 = silent).

        Returns
        -------
        dict  {params, cost_history}
        """
        params = np.random.uniform(0, 2 * np.pi, (2, p), requires_grad=True)
        self.cost_history = []

        for epoch in range(epochs):
            params = self.opt.step(self.cost_function, params)
            cost = float(self.cost_function(params))
            self.cost_history.append(cost)

            if verbose_every and (epoch + 1) % verbose_every == 0:
                print(f"Epoch {epoch + 1:4d} | Cost: {cost:.5f}")

        self.optimal_params = params
        return {"params": params, "cost_history": self.cost_history}

    # ------------------------------------------------------------------
    # Solution extraction
    # ------------------------------------------------------------------

    def solve(self, shots=1024):
        """
        Sample the optimised circuit and return the best bitstring.

        Returns
        -------
        dict
            solution_bitstring : tuple of ints (e.g. (1, 0, 1, 1))
            bitstring_counts   : Counter mapping bitstring -> frequency
            params             : optimal gamma/beta angles
            cost_history       : list of cost values per epoch
        """
        if self.optimal_params is None:
            raise RuntimeError("Call fit() before solve().")

        samples = self.model.sample(self.optimal_params, shots=shots)
        bitstrings = [tuple(int(b) for b in row) for row in samples]
        counts = Counter(bitstrings)
        best = counts.most_common(1)[0][0]

        return {
            "solution_bitstring": best,
            "bitstring_counts": counts,
            "params": self.optimal_params,
            "cost_history": self.cost_history,
        }
