"""
End-to-end VQE demo for the H2 molecule.

(a) Single-point ground state at equilibrium bond length
(b) Potential energy surface: energy vs bond length
(c) Classical benchmark comparison (HF vs VQE vs FCI)

Usage
-----
cd experiments/VQC_experiment
python eigensolver_exp.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from eigensolver.ChemistryEnvironment import ChemistryEnvironment
from eigensolver.PhysicsAnsatz import PhysicsAnsatz
from eigensolver.VQEModel import VQEModel
from eigensolver.EnergyMinimizer import EnergyMinimizer
from eigensolver.classicalBaseline import ClassicalBaseline


# ======================================================================
# Utility: correlation recovery metric
# ======================================================================

def correlation_recovery(hf_energy, vqe_energy, fci_energy):
    return (hf_energy - vqe_energy) / (hf_energy - fci_energy)


# ======================================================================
# (a) Single-point ground state
# ======================================================================

def single_point_vqe(bond_length=1.4):

    env = ChemistryEnvironment.hydrogen(bond_length=bond_length)

    print(env)
    print(f"Hartree-Fock state: {env.hf_state}")

    ansatz = PhysicsAnsatz(
        method="double_excitation",
        n_electrons=env.n_electrons,
        n_qubits=env.n_qubits,
        hf_state=env.hf_state,
    )

    model = VQEModel(environment=env, ansatz=ansatz)

    minimizer = EnergyMinimizer(
        model,
        optimizer_type="gd",
        stepsize=0.4
    )

    print("\n" + "=" * 50)
    print(f"VQE for H2 | bond = {bond_length:.2f} Bohr")
    print("=" * 50)

    result = minimizer.fit(
        epochs=100,
        conv_tol=1e-6,
        verbose_every=2
    )

    vqe_energy = result["ground_state_energy"]

    # Classical baseline comparison
    baseline = ClassicalBaseline(env)
    classical_results = baseline.calculate()

    hf_energy = classical_results["hf_energy"]
    fci_energy = classical_results["fci_energy"]

    recovery = correlation_recovery(
        hf_energy,
        vqe_energy,
        fci_energy
    )

    print("\n" + "=" * 50)
    print("Classical Benchmark Comparison")
    print("=" * 50)

    print(f"Hartree-Fock energy : {hf_energy:.8f} Ha")
    print(f"VQE energy          : {vqe_energy:.8f} Ha")
    print(f"FCI energy          : {fci_energy:.8f} Ha")

    print(f"\nCorrelation recovered: {recovery * 100:.2f}%")

    return result


# ======================================================================
# (b) Potential energy surface
# ======================================================================

def potential_energy_surface(bond_lengths=None):

    if bond_lengths is None:
        bond_lengths = np.linspace(0.5, 4.0, 15)

    vqe_energies = []
    hf_energies = []
    fci_energies = []

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

        minimizer = EnergyMinimizer(
            model,
            optimizer_type="gd",
            stepsize=0.4
        )

        from pennylane import numpy as pnp

        hf_energy = float(
            model.forward(
                pnp.zeros(ansatz.get_weight_shape())
            )
        )

        baseline = ClassicalBaseline(env)
        fci_energy = baseline.calculate()["fci_energy"]

        result = minimizer.fit(
            epochs=60,
            conv_tol=1e-6,
            verbose_every=0
        )

        vqe_energy = result["ground_state_energy"]

        hf_energies.append(hf_energy)
        vqe_energies.append(vqe_energy)
        fci_energies.append(fci_energy)

        print(
            f"HF = {hf_energy:.6f} | "
            f"VQE = {vqe_energy:.6f} | "
            f"FCI = {fci_energy:.6f}"
        )

    return (
        bond_lengths,
        np.array(vqe_energies),
        np.array(hf_energies),
        np.array(fci_energies),
    )


# ======================================================================
# Plots
# ======================================================================

def plot_pes(bond_lengths, vqe_energies, hf_energies, fci_energies):

    plt.figure(figsize=(7, 4))

    plt.plot(
        bond_lengths,
        hf_energies,
        "s--",
        label="Hartree-Fock"
    )

    plt.plot(
        bond_lengths,
        vqe_energies,
        "o-",
        label="VQE"
    )

    plt.plot(
        bond_lengths,
        fci_energies,
        "^-",
        label="Exact FCI"
    )

    plt.xlabel("Bond length (Bohr)")
    plt.ylabel("Energy (Hartree)")
    plt.title("H₂ Potential Energy Surface")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            os.path.dirname(__file__),
            "h2_pes.png"
        ),
        dpi=150,
    )

    plt.show()

    print("Saved h2_pes.png")


# ======================================================================
# Main
# ======================================================================

def main():

    print("=" * 60)
    print("Single-point VQE benchmark")
    print("=" * 60)

    result = single_point_vqe(1.4)

    print("\n" + "=" * 60)
    print("Potential energy surface")
    print("=" * 60)

    (
        bond_lengths,
        vqe_energies,
        hf_energies,
        fci_energies,
    ) = potential_energy_surface()

    plot_pes(
        bond_lengths,
        vqe_energies,
        hf_energies,
        fci_energies,
    )


if __name__ == "__main__":
    main()