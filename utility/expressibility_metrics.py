import torch
import numpy as np



class ExpressibilityMetric:
    """
    Measures KL divergence between:

        circuit fidelity distribution
        Haar-random fidelity distribution
    """

    def __init__(
        self,
        circuit_fn,
        param_shape,
        n_qubits,
        n_samples=200,
        n_bins=75,
        eps=1e-12,
        seed=42,
    ):
        """
        circuit_fn : callable
            function(params) -> statevector

        param_shape : tuple
            shape of parameter tensor

        n_qubits : int
            number of qubits

        n_samples : int
            number of parameter samples

        n_bins : int
            histogram bins

        eps : float
            numerical stability constant

        seed : int
            reproducibility
        """

        self.circuit_fn = circuit_fn
        self.param_shape = param_shape
        self.n_qubits = n_qubits

        self.n_samples = n_samples
        self.n_bins = n_bins

        self.eps = eps

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.states = None
        self.fidelities = None


    def sample_parameters(self):
        """
        Uniform parameter sampling from:

            [0, 2π]

        Returns:
        list[np.ndarray]
        """

        params = []

        for _ in range(self.n_samples):

            theta = np.random.uniform(
                0,
                2 * np.pi,
                size=self.param_shape,
            )

            params.append(theta)

        return params



    def generate_states(self, params_list):
        """
        Generate quantum states from sampled parameters
        """

        states = []

        for params in params_list:

            state = self.circuit_fn(params)

            state = np.asarray(state)

            states.append(state)

        self.states = states

    def compute_pairwise_fidelities(self):
        """
        Compute:

            |⟨ψ_i | ψ_j⟩|^2

        for random pairs
        """

        fidelities = []

        n_states = len(self.states)

        for i in range(n_states):

            psi_i = self.states[i]

            for j in range(i + 1, n_states):

                psi_j = self.states[j]

                fidelity = np.abs(
                    np.vdot(psi_i, psi_j)
                ) ** 2

                fidelities.append(fidelity)

        self.fidelities = np.array(fidelities)


    def compute_histogram(self, values):
        """
        Convert fidelity samples into probability histogram
        """

        hist, bin_edges = np.histogram(
            values,
            bins=self.n_bins,
            range=(0, 1),
            density=True,
        )

        hist = hist + self.eps

        hist = hist / np.sum(hist)

        return hist, bin_edges


    def haar_fidelity_distribution(self, bin_edges):
        """
        Analytical Haar fidelity distribution:

            P(F) = (N-1)(1-F)^(N-2)

        """

        dim = 2 ** self.n_qubits

        haar_probs = []

        for i in range(len(bin_edges) - 1):

            f_left = bin_edges[i]
            f_right = bin_edges[i + 1]

            mid = 0.5 * (f_left + f_right)

            prob = (dim - 1) * (1 - mid) ** (dim - 2)

            haar_probs.append(prob)

        haar_probs = np.array(haar_probs)

        haar_probs += self.eps

        haar_probs /= np.sum(haar_probs)

        return haar_probs



    def kl_divergence(self, p, q):

        return np.sum(
            p * np.log(p / q)
        )


   

    def compute(self):
        """
        Full expressibility pipeline
        """

        
        params = self.sample_parameters()

        self.generate_states(params)

      
        self.compute_pairwise_fidelities()

        
        circuit_hist, bin_edges = self.compute_histogram(
            self.fidelities
        )

        
        haar_hist = self.haar_fidelity_distribution(
            bin_edges
        )

        kl = self.kl_divergence(
            circuit_hist,
            haar_hist,
        )

        expressibility_score = float(
            1.0 / (1.0 + kl)
        )

        return {

            "kl_divergence": float(kl),

            "expressibility_score": expressibility_score,

            "n_samples": self.n_samples,

            "n_bins": self.n_bins,

            "n_qubits": self.n_qubits,
        }