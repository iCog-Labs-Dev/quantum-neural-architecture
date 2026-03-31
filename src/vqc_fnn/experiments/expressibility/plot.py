import numpy as np
import matplotlib.pyplot as plt

from vqc_fnn.experiments.expressibility.expressibility import haar_fidelity_pdf



def plot_fidelity_histogram(
    fidelities: np.ndarray,
    n_qubits: int,
    save_path: str,
):
    """
    Args:
        fidelities: Sampled fidelities
        n_qubits: Number of qubits
        save_path: Output file path
    """
    F = np.linspace(0, 1, 500)
    haar_pdf = haar_fidelity_pdf(F, n_qubits)

    plt.figure(figsize=(6, 4))
    plt.hist(fidelities, bins=75, density=True, alpha=0.6, label="PQC")
    plt.plot(F, haar_pdf, label="Haar", linewidth=2)
    plt.xlabel("Fidelity")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_expressibility_saturation(
    depths,
    expressibilities,
    save_path: str,
):
    """
    Args:
        depths: Circuit depths
        expressibilities: KL values
        save_path: Output file path
    """
    plt.figure(figsize=(6, 4))
    plt.plot(depths, expressibilities, marker="o")
    plt.xlabel("Circuit depth")
    plt.ylabel("KL divergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
