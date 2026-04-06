"""
End-to-end VQE demo for the H2 molecule.

(a) Single-point ground state at equilibrium bond length.
(b) Potential energy surface: energy vs bond length.

Usage
-----
    cd src/eigensolver
    python run_vqe.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt


from eigensolver.ChemistryEnvironment import ChemistryEnvironment
from eigensolver.PhysicsAnsatz import PhysicsAnsatz
from eigensolver.VQEModel import VQEModel
from eigensolver.EnergyMinimizer import EnergyMinimizer


# ======================================================================
# (a)  Single-point ground state
# ======================================================================

def single_point_vqe(bond_length=1.4):
    """Run VQE for H2 at one bond length and return the result."""
    env = ChemistryEnvironment.hydrogen(bond_length=bond_length)
    print(env)
    print(f"Hamiltonian:\n  {env.hamiltonian}")
    print(f"Hartree-Fock state: {env.hf_state}")

    ansatz = PhysicsAnsatz(
        method="double_excitation",
        n_electrons=env.n_electrons,
        n_qubits=env.n_qubits,
        hf_state=env.hf_state,
    )

    model = VQEModel(environment=env, ansatz=ansatz, device_name="default.qubit")

    minimizer = EnergyMinimizer(model, optimizer_type="gd", stepsize=0.4)

    print(f"\n{'='*50}")
    print(f"VQE for H2  |  bond = {bond_length:.2f} Bohr  |  qubits = {env.n_qubits}")
    print(f"{'='*50}")
    result = minimizer.fit(epochs=100, conv_tol=1e-6, verbose_every=2)
    return result


# ======================================================================
# (b)  Potential energy surface
# ======================================================================

def potential_energy_surface(bond_lengths=None):
    """
    Sweep bond lengths, run VQE at each, and collect energies.
    Also records the Hartree-Fock energy (params = 0) for comparison.
    """
    if bond_lengths is None:
        bond_lengths = np.linspace(0.5, 4.0, 15)

    vqe_energies = []
    hf_energies = []

    for d in bond_lengths:
        print(f"\n--- Bond length = {d:.2f} Bohr ---")
        env = ChemistryEnvironment.hydrogen(bond_length=d)

        ansatz = PhysicsAnsatz(
            method="double_excitation",
            n_electrons=env.n_electrons,
            n_qubits=env.n_qubits,
            hf_state=env.hf_state,
        )
        model = VQEModel(environment=env, ansatz=ansatz)
        minimizer = EnergyMinimizer(model, optimizer_type="gd", stepsize=0.4)

        from pennylane import numpy as pnp
        hf_energy = float(model.forward(pnp.zeros(ansatz.get_weight_shape())))
        hf_energies.append(hf_energy)

        result = minimizer.fit(epochs=60, conv_tol=1e-6, verbose_every=0)
        vqe_energies.append(result["ground_state_energy"])
        print(f"  HF = {hf_energy:.6f}  |  VQE = {result['ground_state_energy']:.6f} Ha")

    return bond_lengths, np.array(vqe_energies), np.array(hf_energies)


# ======================================================================
# Plots
# ======================================================================

def plot_convergence(energy_history):
    plt.figure(figsize=(7, 4))
    plt.plot(energy_history, "o-", markersize=4)
    plt.xlabel("Optimisation step")
    plt.ylabel("Energy (Hartree)")
    plt.title("VQE Convergence for H$_2$")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "vqe_convergence.png"), dpi=150)
    plt.show()
    print("Saved vqe_convergence.png")


def plot_pes(bond_lengths, vqe_energies, hf_energies):
    plt.figure(figsize=(7, 4))
    plt.plot(bond_lengths, hf_energies, "s--", label="Hartree-Fock", markersize=5)
    plt.plot(bond_lengths, vqe_energies, "o-", label="VQE", markersize=5)
    plt.xlabel("Bond length (Bohr)")
    plt.ylabel("Energy (Hartree)")
    plt.title("H$_2$ Potential Energy Surface")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "h2_pes.png"), dpi=150)
    plt.show()
    print("Saved h2_pes.png")


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 60)
    print("  (a) Single-point VQE at equilibrium")
    print("=" * 60)
    result = single_point_vqe(bond_length=1.4)
    plot_convergence(result["energy_history"])

    print("\n" + "=" * 60)
    print("  (b) Potential energy surface")
    print("=" * 60)
    bond_lengths, vqe_energies, hf_energies = potential_energy_surface()
    plot_pes(bond_lengths, vqe_energies, hf_energies)

    eq_idx = np.argmin(vqe_energies)
    print(f"\nEquilibrium bond length: {bond_lengths[eq_idx]:.2f} Bohr")
    print(f"Minimum VQE energy:     {vqe_energies[eq_idx]:.6f} Ha")


if __name__ == "__main__":
    main()
