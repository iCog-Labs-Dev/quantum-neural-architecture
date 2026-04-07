import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import sys
import os

# Ensure utility is importable regardless of working directory
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if _root not in sys.path:
    sys.path.insert(0, _root)


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

        p_circuit_safe = p_circuit + eps
        kl = pnp.sum( 
            self.target[self._mask] * (self._log_target[self._mask] - pnp.log(p_circuit_safe[self._mask]))
        )
        return kl

    def compute_fisher_metrics(self, params):
        """
        Fisher Information metrics via parameter-shift on the KL cost.
        Evaluates gradients at n_params random perturbations around current
        params — gives diverse gradient vectors for a full-rank Fisher matrix.
        Cost: 2 * n_params^2 QNode calls (called only at verbose steps).
        """
        import torch
        from utility.generalization_metrics import FisherGeneralizationMetric

        metric = FisherGeneralizationMetric()
        w = np.array(params).flatten()
        shift = np.pi / 2.0
        n = len(w)
        rng = np.random.default_rng(42)

        # Sample n random points near current params and compute gradient at each.
        # This gives N=n diverse gradient vectors → proper Fisher matrix spectrum.
        noise_scale = 0.1  # small enough to stay in local landscape
        for _ in range(n):
            w_perturbed = w + rng.normal(0, noise_scale, size=n)
            grad = np.zeros(n)
            for k in range(n):
                wp = w_perturbed.copy(); wp[k] += shift
                wm = w_perturbed.copy(); wm[k] -= shift
                grad[k] = (
                    float(self.cost_function(wp.reshape(params.shape))) -
                    float(self.cost_function(wm.reshape(params.shape)))
                ) / 2.0
            metric.accumulate(torch.from_numpy(grad.astype(np.float64)))

        return metric.compute()

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
        fisher_history = []

        for epoch in range(1, epochs + 1):
            params = self.optimizer.step(self.cost_function, params)
            cost = float(self.cost_function(params))
            cost_history.append(cost)

            if verbose_every and epoch % verbose_every == 0:
                fisher = self.compute_fisher_metrics(params)
                fisher_history.append(fisher)
                ed = fisher['effective_dimension']
                se = fisher['spectral_entropy_normalized']
                gb = fisher['generalization_bound']
                print(f"  Epoch {epoch:4d} | KL: {cost:.6f} | d_eff: {ed:.2f} | Entropy: {se:.3f} | Gen bound: {gb:.4f}")

            if len(cost_history) >= 2 and abs(cost_history[-2] - cost) < conv_tol:
                if verbose_every:
                    fisher = self.compute_fisher_metrics(params)
                    fisher_history.append(fisher)
                    ed = fisher['effective_dimension']
                    se = fisher['spectral_entropy_normalized']
                    gb = fisher['generalization_bound']
                    print(f"  Converged at epoch {epoch} (delta < {conv_tol}) | d_eff: {ed:.2f} | Entropy: {se:.3f} | Gen bound: {gb:.4f}")
                break

        final_dist = np.array(self.model.forward(params))

        return {
            "params": params,
            "cost_history": cost_history,
            "fisher_history": fisher_history,
            "final_distribution": final_dist,
        }
