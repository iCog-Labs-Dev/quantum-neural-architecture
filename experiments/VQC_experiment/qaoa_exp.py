import os
from utility import FIGURES_DIR, MODELS_DIR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from QAOA.context import Context
from QAOA.ProblemFormulator import ProblemFormulator
from QAOA.StatePreparation import StatePreparation
from QAOA.QAOAAnsatz import QAOAAnsatz
from QAOA.QAOAModel import QAOAModel
from QAOA.QAOAOptimizer import QAOAOptimizer


def build_example_context():
    """
    A toy binary context (objects x attributes).
    Rows = objects, Columns = attributes, 1 = object has attribute.
    """
    data = pd.DataFrame(
        {
            "a": [1, 1, 0, 0],
            "b": [1, 0, 1, 0],
            "c": [0, 1, 1, 1],
            "d": [0, 0, 1, 1],
        },
        index=["o1", "o2", "o3", "o4"],
    )
    return Context(data=data)


# ======================================================================
# 2.  Extract concepts and build QUBO
# ======================================================================

def build_qubo(context):
    lattice = context.extract_concepts()

    concepts = lattice.get_concept_lattice()
    print(f"Extracted {len(concepts)} concepts:")
    for i, c in enumerate(concepts):
        print(f"  x_{i}: Extent={c.get_extent()}, Intent={c.get_intent()}")

    Q = lattice.set_cover()
    print(f"\nQUBO matrix Q ({Q.shape[0]}x{Q.shape[1]}):")
    print(Q)
    return Q, lattice


# ======================================================================
# 3.  Run QAOA
# ======================================================================

def run_qaoa(Q, p=2, epochs=100, stepsize=0.3):
    n_qubits = Q.shape[0]

    formulator = ProblemFormulator(Q)
    cost_h = formulator.get_cost_hamiltonian()
    mixer_h = formulator.get_mixer_hamiltonian()
    offset = formulator.get_offset()
    print(f"\nIsing offset (constant shift): {offset:.4f}")
    print(f"Cost Hamiltonian:\n  {cost_h}")
    print(f"Mixer Hamiltonian:\n  {mixer_h}")

    ansatz = QAOAAnsatz(cost_h, mixer_h, p=p)
    model = QAOAModel(
        n_qubits=n_qubits,
        qaoa_ansatz=ansatz,
        cost_hamiltonian=cost_h,
    )

    optimizer = QAOAOptimizer(model, optimizer_type="adam", stepsize=stepsize)

    print(f"\n{'='*50}")
    print(f"Running QAOA  |  qubits={n_qubits}  p={p}  epochs={epochs}")
    print(f"{'='*50}")
    fit_result = optimizer.fit(p=p, epochs=epochs, verbose_every=20)

    result = optimizer.solve(shots=1024)
    return result, offset


# ======================================================================
# 4.  Interpret solution
# ======================================================================

def interpret(result, offset, lattice):
    bitstring = result["solution_bitstring"]
    counts = result["bitstring_counts"]

    print(f"\n{'='*50}")
    print("QAOA Result")
    print(f"{'='*50}")
    print(f"Most probable bitstring: {bitstring}")
    print(f"Top-5 bitstrings by frequency:")
    for bs, freq in counts.most_common(5):
        print(f"  {bs}  ->  {freq} counts")

    concepts = lattice.get_concept_lattice()
    selected = [i for i, bit in enumerate(bitstring) if bit == 1]
    print(f"\nSelected concepts (x_i = 1):")
    for i in selected:
        c = concepts[i]
        print(f"  Concept {i}: Extent={c.get_extent()}, Intent={c.get_intent()}")

    return selected


# ======================================================================
# 5.  Plot convergence
# ======================================================================

def plot_convergence(cost_history, offset):
    plt.figure(figsize=(8, 4))
    adjusted = [c + offset for c in cost_history]
    plt.plot(cost_history, label="<H_cost> (circuit)")
    plt.plot(adjusted, label="QUBO objective (with offset)", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Energy")
    plt.title("QAOA Convergence")
    plt.legend()
    plt.tight_layout()
    out_path = FIGURES_DIR / "qaoa_convergence.png"
    os.makedirs(out_path.parent, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Figure saved to {out_path}")


# ======================================================================
# Main
# ======================================================================

def main():
    context = build_example_context()
    print(context)

    Q, lattice = build_qubo(context)

    result, offset = run_qaoa(Q, p=2, epochs=100, stepsize=0.3)

    selected = interpret(result, offset, lattice)

    import torch
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({"solution": result["solution_bitstring"], "params": result.get("params")},
               MODELS_DIR / "qaoa_solution.pt")
    print(f"Model saved to {MODELS_DIR / 'qaoa_solution.pt'}")

    plot_convergence(result["cost_history"], offset)


if __name__ == "__main__":
    main()
