"""
Cyclical time-series proof: VQC vs classical FFNN.

A synthetic binary classification task where the positive class wraps
around midnight (hours ~21-05).  The VQC cyclical embedding maps the hour
to an RZ rotation whose 2*pi periodicity handles the wrap-around natively.
A plain FFNN that takes the raw hour scalar sees 0 and 23 as maximally
distant -- it needs explicit sin/cos feature engineering to compete.

We compare three models:
  1. VQC with cyclical embedding  (1 qubit, RZ)
  2. Classical FFNN on raw hour   (1 input neuron)
  3. Classical FFNN on sin/cos    (2 input neurons -- hand-engineered)

Usage
-----
    python experiment_cyclical.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))

from pennylane import numpy as pnp

from Embedding import EmbeddingLayer
from Ansatz import AnsatzLayer
from VQCModel import VQCModel
from Optimizer import Trainer
from ClassicalBaseline import ClassicalBaseline
from data_utils import to_cyclical, generate_cyclical_dataset


# ======================================================================
# 1.  Generate data
# ======================================================================

def load_data(n_samples=300, seed=42):
    hours, labels = generate_cyclical_dataset(n_samples=n_samples, seed=seed)

    angles, sin_cos = to_cyclical(hours, period=24.0)

    h_train, h_test, y_train, y_test = train_test_split(
        hours, labels, test_size=0.25, random_state=seed, stratify=labels
    )
    a_train, a_test, _, _ = train_test_split(
        angles, labels, test_size=0.25, random_state=seed, stratify=labels
    )
    sc_train, sc_test, _, _ = train_test_split(
        sin_cos, labels, test_size=0.25, random_state=seed, stratify=labels
    )

    return (h_train, h_test, a_train, a_test, sc_train, sc_test,
            y_train, y_test, hours, labels, angles)


# ======================================================================
# 2.  Train VQC  (cyclical embedding, 1 qubit)
# ======================================================================

def train_vqc(a_train, y_train, a_test, y_test,
              n_layers=4, epochs=40, stepsize=0.15):
    n_qubits = 1

    embedding = EmbeddingLayer(method="cyclical")
    ansatz = AnsatzLayer(method="strong", n_layers=n_layers)
    model = VQCModel(
        n_qubits=n_qubits,
        embedding=embedding,
        ansatz=ansatz,
        measurement="expval",
        dropout_rate=0.0,
    )

    X_tr = pnp.array(a_train.reshape(-1, 1), requires_grad=False)
    Y_tr = pnp.array(y_train, requires_grad=False)
    X_te = pnp.array(a_test.reshape(-1, 1), requires_grad=False)
    Y_te = pnp.array(y_test, requires_grad=False)

    trainer = Trainer(model, optimizer_type="adam", stepsize=stepsize)

    print("\n=== VQC Training (cyclical embedding, 1 qubit) ===")
    results = trainer.fit(
        X_tr, Y_tr, epochs=epochs,
        X_val=X_te, Y_val=Y_te,
        verbose_every=5,
    )

    weights = results["weights"]
    model.eval()

    preds_raw = np.array([float(model.forward(x, weights)) for x in X_te])
    preds_class = (preds_raw > 0.0).astype(int)
    acc = np.mean(preds_class == y_test.astype(int))
    print(f"VQC test accuracy: {acc:.2%}")

    return results, acc


# ======================================================================
# 3.  Train Classical FFNNs
# ======================================================================

def train_classical_raw(h_train, y_train, h_test, y_test, epochs=400):
    """FFNN on raw hour scalar (1 input)."""
    X_tr = h_train.reshape(-1, 1).astype(np.float32) / 24.0
    X_te = h_test.reshape(-1, 1).astype(np.float32) / 24.0

    baseline = ClassicalBaseline(n_input=1, hidden_size=16, lr=0.01)
    print(f"\n=== FFNN on raw hour ({baseline.param_count()} params) ===")
    res = baseline.fit(
        X_tr, y_train.astype(np.float32), epochs=epochs,
        X_val=X_te, Y_val=y_test.astype(np.float32),
        verbose_every=100,
    )
    preds = baseline.predict_classes(X_te)
    acc = np.mean(preds == y_test.astype(int))
    print(f"FFNN (raw) test accuracy: {acc:.2%}")
    return res, acc


def train_classical_sincos(sc_train, y_train, sc_test, y_test, epochs=400):
    """FFNN on sin/cos-encoded hour (2 inputs)."""
    X_tr = sc_train.astype(np.float32)
    X_te = sc_test.astype(np.float32)

    baseline = ClassicalBaseline(n_input=2, hidden_size=16, lr=0.01)
    print(f"\n=== FFNN on sin/cos ({baseline.param_count()} params) ===")
    res = baseline.fit(
        X_tr, y_train.astype(np.float32), epochs=epochs,
        X_val=X_te, Y_val=y_test.astype(np.float32),
        verbose_every=100,
    )
    preds = baseline.predict_classes(X_te)
    acc = np.mean(preds == y_test.astype(int))
    print(f"FFNN (sin/cos) test accuracy: {acc:.2%}")
    return res, acc


# ======================================================================
# 4.  Plots
# ======================================================================

def plot_results(hours, labels, vqc_res, raw_res, sc_res,
                 vqc_acc, raw_acc, sc_acc):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    # (a) Dataset
    ax = axes[0]
    colours = ["salmon" if l == 0 else "steelblue" for l in labels]
    ax.scatter(hours, labels + np.random.normal(0, 0.03, len(labels)),
               c=colours, s=15, alpha=0.6, edgecolors="k", linewidths=0.3)
    ax.axvline(5, color="grey", linestyle="--", alpha=0.5)
    ax.axvline(21, color="grey", linestyle="--", alpha=0.5)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Class (jittered)")
    ax.set_title("Cyclical dataset (boundary wraps at 0/24)")
    ax.set_xlim(-0.5, 24.5)

    # (b) Training curves
    ax = axes[1]
    ax.plot(vqc_res["train_history"], label="VQC train")
    if vqc_res["val_history"]:
        ax.plot(vqc_res["val_history"], label="VQC val", linestyle="--")
    ax.plot(raw_res["train_history"], label="FFNN raw train", alpha=0.6)
    ax.plot(sc_res["train_history"], label="FFNN sin/cos train", alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cost / Loss")
    ax.set_title("Training curves")
    ax.legend(fontsize=8)

    # (c) Accuracy comparison
    ax = axes[2]
    names = ["VQC\ncyclical", "FFNN\nraw hour", "FFNN\nsin/cos"]
    accs = [vqc_acc, raw_acc, sc_acc]
    colors = ["steelblue", "salmon", "orange"]
    bars = ax.bar(names, accs, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy comparison")
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.1%}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "cyclical_results.png"), dpi=150)
    plt.show()
    print("Figure saved to cyclical_results.png")


# ======================================================================
# 5.  Main
# ======================================================================

def main():
    (h_train, h_test, a_train, a_test, sc_train, sc_test,
     y_train, y_test, hours, labels, angles) = load_data()

    vqc_res, vqc_acc = train_vqc(a_train, y_train, a_test, y_test)
    raw_res, raw_acc = train_classical_raw(h_train, y_train, h_test, y_test)
    sc_res, sc_acc = train_classical_sincos(sc_train, y_train, sc_test, y_test)

    print(f"\n{'='*50}")
    print(f"VQC (cyclical RZ)         accuracy: {vqc_acc:.2%}")
    print(f"FFNN (raw hour)           accuracy: {raw_acc:.2%}")
    print(f"FFNN (sin/cos engineered) accuracy: {sc_acc:.2%}")
    print(f"{'='*50}")

    plot_results(hours, labels, vqc_res, raw_res, sc_res,
                 vqc_acc, raw_acc, sc_acc)


if __name__ == "__main__":
    main()
