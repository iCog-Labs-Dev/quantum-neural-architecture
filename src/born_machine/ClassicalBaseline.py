import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class RestrictedBoltzmannMachine:
    """
    Binary-binary RBM for generative modelling over bitstrings.

    Parameters
    ----------
    n_visible : int
        Number of visible units (= number of qubits / bits).
    n_hidden : int
        Number of hidden units.  Choose so that the total parameter
        count (n_vis * n_hid + n_vis + n_hid) roughly matches the
        Born Machine's parameter count.
    seed : int or None
    """

    def __init__(self, n_visible, n_hidden, seed=42):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        rng = np.random.default_rng(seed)

        self.W = rng.normal(0, 0.01, size=(n_visible, n_hidden))
        self.b_vis = np.zeros(n_visible)
        self.b_hid = np.zeros(n_hidden)

    def param_count(self):
        return self.n_visible * self.n_hidden + self.n_visible + self.n_hidden

    # ----- conditional distributions -----

    def _prob_hidden_given_visible(self, v):
        """P(h_j = 1 | v) = sigmoid(c_j + sum_i v_i W_ij)"""
        return _sigmoid(self.b_hid + v @ self.W)

    def _prob_visible_given_hidden(self, h):
        """P(v_i = 1 | h) = sigmoid(b_i + sum_j W_ij h_j)"""
        return _sigmoid(self.b_vis + h @ self.W.T)

    def _sample_hidden(self, v, rng):
        prob = self._prob_hidden_given_visible(v)
        return (rng.random(size=prob.shape) < prob).astype(float)

    def _sample_visible(self, h, rng):
        prob = self._prob_visible_given_hidden(h)
        return (rng.random(size=prob.shape) < prob).astype(float)

    # ----- energy & probabilities -----

    def _free_energy(self, v):
        """F(v) = -b^T v - sum_j log(1 + exp(c_j + sum_i v_i W_ij))"""
        wx = v @ self.W + self.b_hid
        return -v @ self.b_vis - np.sum(np.log(1.0 + np.exp(np.clip(wx, -500, 500))))

    def probabilities(self):
        """
        Brute-force P(v) for all 2^n_visible bitstrings.
        Feasible only for small n_visible.
        """
        n = self.n_visible
        n_states = 2 ** n
        energies = np.zeros(n_states)
        for idx in range(n_states):
            bits = np.array(
                [int(b) for b in format(idx, f"0{n}b")], dtype=float
            )
            energies[idx] = -self._free_energy(bits)

        log_z = np.max(-energies) + np.log(np.sum(np.exp(-energies - np.max(-energies))))
        log_probs = -energies - log_z
        return np.exp(log_probs)

    # ----- Contrastive Divergence training -----

    def fit(self, data_samples, epochs=500, lr=0.01, k=1,
            verbose_every=50, target_distribution=None):
        """
        Train via CD-k.

        Parameters
        ----------
        data_samples : ndarray of shape (n_samples, n_visible)
            Binary training vectors.
        epochs : int
        lr : float
        k : int
            Number of Gibbs steps in CD.
        verbose_every : int
        target_distribution : ndarray or None
            If given, KL divergence is tracked each epoch for comparison.

        Returns
        -------
        dict with keys: cost_history (KL values or empty)
        """
        rng = np.random.default_rng(0)
        n_samples = data_samples.shape[0]
        cost_history = []

        for epoch in range(1, epochs + 1):
            order = rng.permutation(n_samples)
            for s in order:
                v0 = data_samples[s]
                h0 = self._sample_hidden(v0, rng)

                vk = v0.copy()
                for _ in range(k):
                    hk = self._sample_hidden(vk, rng)
                    vk = self._sample_visible(hk, rng)
                hk_prob = self._prob_hidden_given_visible(vk)

                positive = np.outer(v0, self._prob_hidden_given_visible(v0))
                negative = np.outer(vk, hk_prob)

                self.W += lr * (positive - negative)
                self.b_vis += lr * (v0 - vk)
                self.b_hid += lr * (
                    self._prob_hidden_given_visible(v0) - hk_prob
                )

            if target_distribution is not None:
                kl = self._kl_divergence(target_distribution)
                cost_history.append(kl)
                if verbose_every and epoch % verbose_every == 0:
                    print(f"  Epoch {epoch:4d} | KL divergence: {kl:.6f}")

        return {"cost_history": cost_history}

    def _kl_divergence(self, p_target):
        """KL(p_target || p_model), only over entries where p_target > 0."""
        p_model = self.probabilities()
        eps = 1e-10
        p_model_safe = np.clip(p_model, eps, None)
        mask = p_target > 0
        return float(np.sum(
            p_target[mask] * (np.log(p_target[mask]) - np.log(p_model_safe[mask]))
        ))

    # ----- sampling -----

    def generate_samples(self, n_samples, gibbs_steps=100, seed=0):
        """Run Gibbs chains to produce samples."""
        rng = np.random.default_rng(seed)
        v = (rng.random(size=(n_samples, self.n_visible)) > 0.5).astype(float)

        for _ in range(gibbs_steps):
            h = self._sample_hidden(v, rng)
            v = self._sample_visible(h, rng)

        return v
