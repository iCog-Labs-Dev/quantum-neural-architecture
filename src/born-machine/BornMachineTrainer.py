"""
KL divergence optimiser for the Born Machine.

No X, no Y, no MSE. The cost is the divergence between the circuit's
probability vector and the target distribution.
"""

import numpy as np
import pennylane as qml


class BornMachineTrainer:
    """
    Trains a BornMachineModel by minimising the KL divergence
    between the target distribution and the circuit's Born distribution.
    """

    SUPPORTED_OPTIMIZERS = ("adam", "gradient_descent", "nesterov", "spsa")

    def __init__(
        self,
        model,
        target_distribution,
        optimizer_type="adam",
        stepsize=0.1,
    ):
        self.model = model
        self.target = np.array(target_distribution, dtype=float)
        self.optimizer_type = optimizer_type.lower()
        self.stepsize = stepsize

        self._mask = self.target > 0
        self._log_target = np.zeros_like(self.target)
        self._log_target[self._mask] = np.log(self.target[self._mask])

        self.optimizer = self._build_optimizer()

    def _build_optimizer(self):
        otype = self.optimizer_type
        lr = self.stepsize
        if otype == "adam":
            return qml.AdamOptimizer(stepsize=lr)
        elif otype == "gradient_descent":
            return qml.GradientDescentOptimizer(stepsize=lr)
        elif otype == "nesterov":
            return qml.NesterovMomentumOptimizer(stepsize=lr)
        elif otype == "spsa":
            return qml.SPSAOptimizer(maxiter=200)
        else:
            raise ValueError(
                f"Unsupported optimizer: '{otype}'. "
                f"Choose from {self.SUPPORTED_OPTIMIZERS}"
            )

    def cost_function(self, params):
        """
        KL(P_target || P_circuit) = sum_x P_target(x) * log(P_target(x) / P_circuit(x))

        Only summed over entries where P_target > 0.
        A small epsilon avoids log(0) in the circuit distribution.
        """
        p_circuit = self.model.forward(params)
        eps = 1e-10
        p_circuit_safe = np.clip(p_circuit, eps, None)

        kl = np.sum(
            self.target[self._mask]
            * (self._log_target[self._mask] - np.log(p_circuit_safe[self._mask]))
        )
        return kl

    def fit(self, epochs=300, conv_tol=1e-6, verbose_every=50, seed=42):
        """
        Train the Born Machine.

        Parameters
        ----------
        epochs : int
        conv_tol : float
            Stop when consecutive cost changes less than this.
        verbose_every : int
            Print progress every N epochs. 0 to suppress output.
        seed : int or None

        Returns
        -------
        dict with keys: params, cost_history, final_distribution
        """
        params = self.model.init_params(seed=seed)
        cost_history = []

        for epoch in range(1, epochs + 1):
            params = self.optimizer.step(self.cost_function, params)
            cost = float(self.cost_function(params))
            cost_history.append(cost)

            if verbose_every and epoch % verbose_every == 0:
                print(f"  Epoch {epoch:4d} | KL divergence: {cost:.6f}")

            if len(cost_history) >= 2 and abs(cost_history[-2] - cost) < conv_tol:
                if verbose_every:
                    print(f"  Converged at epoch {epoch} (delta < {conv_tol})")
                break

        final_dist = np.array(self.model.forward(params))

        return {
            "params": params,
            "cost_history": cost_history,
            "final_distribution": final_dist,
        }
