import numpy as np
import torch
import pennylane as qml
from itertools import combinations
from scipy.stats import entropy
from typing import List


def haar_fidelity_pdf(F: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Args:
        F: Fidelity values in [0, 1]
        n_qubits: Number of qubits

    Returns:
        Haar-random fidelity probability density
    """
    N = 2 ** n_qubits
    return (N - 1) * (1 - F) ** (N - 2)


def sample_quantum_states(
    vqc,
    num_samples: int,
) -> torch.Tensor:
    """
    Args:
        vqc: VQCLayer instance
        num_samples: Number of states to sample

    Returns:
        Tensor of shape (num_samples, 2**n_qubits)
    """
    states = []
    n_qubits = vqc.n_qubits
    x_dummy = torch.zeros(n_qubits)

    for _ in range(num_samples):
        with torch.no_grad():
            vqc.weights.copy_(torch.randn_like(vqc.weights) * np.pi)

            @qml.qnode(vqc.device)
            def state_circuit():
                if vqc.encoding_strategy == "single":
                    vqc.encoder(x_dummy)
                    vqc.ansatz_fn(vqc.weights)
                else:
                    for layer in range(vqc.n_layers):
                        vqc.encoder(x_dummy)
                        vqc.ansatz_fn(vqc.weights[layer].unsqueeze(0))
                return qml.state()

            state = torch.tensor(state_circuit(), dtype=torch.complex64)
            states.append(state)

    return torch.stack(states)


def compute_fidelities(states: torch.Tensor) -> np.ndarray:
    """
    Args:
        states: Tensor of quantum states (num_states, dim)

    Returns:
        Array of fidelities
    """
    fidelities = []
    for i, j in combinations(range(len(states)), 2):
        overlap = torch.abs(
            torch.dot(torch.conj(states[i]), states[j])
        ) ** 2
        fidelities.append(overlap.item())
    return np.asarray(fidelities)


def compute_expressibility_kl(
    fidelities: np.ndarray,
    n_qubits: int,
    n_bins: int = 75,
) -> float:
    """
    Args:
        fidelities: Sampled fidelities
        n_qubits: Number of qubits
        n_bins: Histogram bins

    Returns:
        KL divergence (expressibility)
    """
    hist_pqc, bin_edges = np.histogram(
        fidelities, bins=n_bins, range=(0, 1), density=True
    )

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist_haar = haar_fidelity_pdf(bin_centers, n_qubits)

    hist_pqc += 1e-10
    hist_haar += 1e-10

    hist_pqc /= hist_pqc.sum()
    hist_haar /= hist_haar.sum()

    return entropy(hist_pqc, hist_haar)
