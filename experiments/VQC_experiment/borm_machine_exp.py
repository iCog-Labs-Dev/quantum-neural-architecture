"""
Bars-and-Stripes benchmark: Born Machine vs Restricted Boltzmann Machine.

End-to-end experiment:
  1. Generate 2x2 BAS target distribution (6 valid patterns, 4 qubits).
  2. Train Born Machine via KL divergence.
  3. Train RBM with matched parameter count via CD-1.
  4. Compare KL convergence, probability mass, and sample quality.
"""

import os


import numpy as np
import matplotlib.pyplot as plt

from born_machine.data_utils import (
    bars_and_stripes,
    bitstring_to_grid,
    index_to_bitstring,
    empirical_distribution,
)
from born_machine.Ansatz import AnsatzLayer
from born_machine.BornMachineModel import BornMachineModel
from born_machine.BornMachineTrainer import BornMachineTrainer
from born_machine.ClassicalBaseline import RestrictedBoltzmannMachine


# ────────────────────────── helpers ──────────────────────────

def _valid_mass(probs, valid_indices):
    return float(np.sum(probs[valid_indices]))


def _invalid_mass(probs, valid_indices):
    return 1.0 - _valid_mass(probs, valid_indices)


# ────────────────────────── configuration ──────────────────────────

GRID_SIZE = 2
N_QUBITS = GRID_SIZE * GRID_SIZE       # 4
N_LAYERS = 6
BORN_EPOCHS = 300
RBM_EPOCHS = 500
BORN_LR = 0.1
RBM_LR = 0.01
SEED = 42


def main():
    print("=" * 60)
    print("  Bars-and-Stripes Benchmark: Born Machine vs RBM")
    print("=" * 60)

    # 1. Target distribution
    target, valid_idx = bars_and_stripes(GRID_SIZE)
    n_valid = len(valid_idx)
    n_bits = N_QUBITS
    print(f"\nGrid: {GRID_SIZE}x{GRID_SIZE}  |  Qubits: {n_bits}")
    print(f"Valid BAS patterns: {n_valid} / {2**n_bits}")
    print("Valid indices:", valid_idx)
    for vi in valid_idx:
        bits = index_to_bitstring(vi, n_bits)
        print(f"  {bits}  ->  ", bitstring_to_grid(bits, GRID_SIZE).tolist())

    # 2. Born Machine
    print("\n" + "-" * 60)
    print("Training Born Machine")
    print("-" * 60)

    ansatz = AnsatzLayer(method="strong", n_layers=N_LAYERS)
    model = BornMachineModel(n_qubits=N_QUBITS, ansatz=ansatz)
    born_params = model.param_count()
    print(f"  Parameters: {born_params}")

    trainer = BornMachineTrainer(
        model, target, optimizer_type="adam", stepsize=BORN_LR
    )
    result_born = trainer.fit(
        epochs=BORN_EPOCHS, conv_tol=1e-7, verbose_every=50, seed=SEED
    )
    born_dist = result_born["final_distribution"]

    # Save Born Machine params
    import torch
    model_path = os.path.join(project_root, "results", "models", "born_machine.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({"params": result_born["params"], "final_kl": result_born["cost_history"][-1]}, model_path)
    print(f"Model saved to {model_path}")

    # 3. RBM with matched parameter count
    print("\n" + "-" * 60)
    print("Training Restricted Boltzmann Machine")
    print("-" * 60)

    n_hidden = max(1, round((born_params - N_QUBITS) / (N_QUBITS + 1)))
    rbm = RestrictedBoltzmannMachine(
        n_visible=N_QUBITS, n_hidden=n_hidden, seed=SEED
    )
    rbm_params = rbm.param_count()
    print(f"  Parameters: {rbm_params}  (hidden units: {n_hidden})")

    valid_samples = []
    for vi in valid_idx:
        valid_samples.append(
            np.array([int(b) for b in format(vi, f"0{n_bits}b")], dtype=float)
        )
    train_data = np.array(valid_samples)

    result_rbm = rbm.fit(
        data_samples=train_data,
        epochs=RBM_EPOCHS,
        lr=RBM_LR,
        k=1,
        verbose_every=50,
        target_distribution=target,
    )
    rbm_dist = rbm.probabilities()

    # 4. Summary
    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)

    print(f"\n{'Model':<20} {'Params':>8} {'Final KL':>12} "
          f"{'Valid Mass':>12} {'Invalid Mass':>14}")
    print("-" * 70)

    born_kl = result_born["cost_history"][-1]
    rbm_kl = result_rbm["cost_history"][-1] if result_rbm["cost_history"] else float("nan")

    for label, dist, kl, pc in [
        ("Born Machine", born_dist, born_kl, born_params),
        ("RBM (CD-1)", rbm_dist, rbm_kl, rbm_params),
    ]:
        vm = _valid_mass(dist, valid_idx)
        im = _invalid_mass(dist, valid_idx)
        print(f"{label:<20} {pc:>8} {kl:>12.6f} {vm:>12.4f} {im:>14.4f}")

    # 5. Plots
    _plot_kl_convergence(result_born["cost_history"], result_rbm["cost_history"])
    _plot_distributions(target, born_dist, rbm_dist, valid_idx, n_bits)
    _plot_sample_grids(result_born["params"], model, rbm, GRID_SIZE)


# ────────────────────────── plotting ──────────────────────────

def _plot_kl_convergence(born_history, rbm_history):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(born_history, label="Born Machine", linewidth=2)
    if rbm_history:
        ax.plot(rbm_history, label="RBM (CD-1)", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence Convergence")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, "results", "figures", "kl_convergence.png"), dpi=150)
    plt.show()


def _plot_distributions(target, born_dist, rbm_dist, valid_idx, n_bits):
    labels = [format(i, f"0{n_bits}b") for i in range(2**n_bits)]
    x = np.arange(2**n_bits)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, target, width, label="Target", color="black", alpha=0.7)
    ax.bar(x, born_dist, width, label="Born Machine", color="tab:blue", alpha=0.7)
    ax.bar(x + width, rbm_dist, width, label="RBM", color="tab:orange", alpha=0.7)

    for vi in valid_idx:
        ax.axvline(vi, color="green", linewidth=0.5, alpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_xlabel("Bitstring")
    ax.set_ylabel("Probability")
    ax.set_title("Learned Distributions vs Target (2x2 Bars-and-Stripes)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, "results", "figures", "distributions.png"), dpi=150)
    plt.show()


def _plot_sample_grids(born_params, model, rbm, grid_size):
    n_show = 8
    born_samples = model.sample(born_params, shots=n_show)
    rbm_samples = rbm.generate_samples(n_show, gibbs_steps=200, seed=99)

    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 1.5, 3.5))
    fig.suptitle("Generated Samples", fontsize=13)

    for i in range(n_show):
        grid_b = bitstring_to_grid(born_samples[i], grid_size)
        axes[0, i].imshow(grid_b, cmap="binary", vmin=0, vmax=1)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        if i == 0:
            axes[0, i].set_ylabel("Born\nMachine", fontsize=10)

        grid_r = bitstring_to_grid(rbm_samples[i], grid_size)
        axes[1, i].imshow(grid_r, cmap="binary", vmin=0, vmax=1)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        if i == 0:
            axes[1, i].set_ylabel("RBM", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(project_root, "results", "figures", "samples.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
