"""
Toy proof: VQC vs classical FFNN on sklearn.make_circles.

The two classes are concentric circles.  A linear FFNN cannot separate them
without heavy feature engineering, whereas the VQC with a spherical embedding
sees the circular geometry natively on the Bloch sphere.

Usage
-----
    python experiment_circles.py
"""

import os
from utility import FIGURES_DIR, MODELS_DIR

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split


from pennylane import numpy as pnp

from non_euclidian.Embedding import EmbeddingLayer
from non_euclidian.Ansatz import AnsatzLayer
from non_euclidian.VQCModel import VQCModel
from non_euclidian.Optimizer import Trainer
from non_euclidian.ClassicalBaseline import ClassicalBaseline
from non_euclidian.data_utils import to_polar, polar_to_bloch


# ======================================================================
# 1.  Generate data
# ======================================================================

def load_data(n_samples=200, noise=0.1, seed=42):
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.4, random_state=seed)
    y = y.astype(float)

    X_polar = to_polar(X)
    X_bloch = polar_to_bloch(X_polar)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        X_bloch, y, test_size=0.25, random_state=seed
    )

    return (X_train, X_test, y_train, y_test,
            Xb_train, Xb_test, yb_train, yb_test,
            X, y, X_bloch)


# ======================================================================
# 2.  Train VQC
# ======================================================================

def train_vqc(Xb_train, yb_train, Xb_test, yb_test,
              n_layers=2, epochs=300, stepsize=0.15):
    n_qubits = 2

    embedding = EmbeddingLayer(method="spherical")
    ansatz = AnsatzLayer(method="strong", n_layers=n_layers)
    model = VQCModel(
        n_qubits = n_qubits,
        embedding = embedding,
        ansatz = ansatz,
        measurement = "expval",
        dropout_rate = 0.0,
    )

    Xb_train_pnp = pnp.array(Xb_train, requires_grad=False)
    yb_train_pnp = pnp.array(yb_train, requires_grad=False)
    Xb_test_pnp = pnp.array(Xb_test, requires_grad=False)
    yb_test_pnp = pnp.array(yb_test, requires_grad=False)

    trainer = Trainer(model, optimizer_type="adam", stepsize=stepsize)

    print("\n=== VQC Training (spherical embedding) ===")
    results = trainer.fit(
        Xb_train_pnp, yb_train_pnp,
        epochs=epochs,
        X_val=Xb_test_pnp, Y_val=yb_test_pnp,
        verbose_every=5,
    )

    weights = results["weights"]
    model.eval()

    preds_raw = np.array([float(model.forward(x, weights)) for x in Xb_test_pnp])
    preds_class = (preds_raw > 0.0).astype(int)
    acc = np.mean(preds_class == yb_test.astype(int))
    print(f"VQC test accuracy: {acc:.2%}")

    return results, model, weights, acc


# ======================================================================
# 3.  Train Classical FFNN
# ======================================================================

def train_classical(X_train, y_train, X_test, y_test, epochs=300):
    baseline = ClassicalBaseline(n_input=2, hidden_size=4, lr=0.01)
    print(f"\n=== Classical FFNN ({baseline.param_count()} params) ===")
    cls_results = baseline.fit(
        X_train.astype(np.float32), y_train.astype(np.float32),
        epochs=epochs,
        X_val=X_test.astype(np.float32), Y_val=y_test.astype(np.float32),
        verbose_every=50,
    )
    preds = baseline.predict_classes(X_test.astype(np.float32))
    acc = np.mean(preds == y_test.astype(int))
    print(f"FFNN test accuracy: {acc:.2%}")
    return cls_results, baseline, acc


# ======================================================================
# 4.  Plots
# ======================================================================

def plot_results(X, y, vqc_results, cls_results, vqc_acc, cls_acc):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (a) Dataset
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k", s=20, alpha=0.7)
    ax.set_title("make_circles dataset")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    # (b) Training loss curves
    ax = axes[1]
    ax.plot(vqc_results["train_history"], label="VQC train")
    if vqc_results["val_history"]:
        ax.plot(vqc_results["val_history"], label="VQC val", linestyle="--")
    ax.plot(cls_results["train_history"], label="FFNN train", alpha=0.7)
    if cls_results["val_history"]:
        ax.plot(cls_results["val_history"], label="FFNN val", linestyle="--", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cost / Loss")
    ax.set_title("Training curves")
    ax.legend()

    # (c) Accuracy bar chart
    ax = axes[2]
    bars = ax.bar(["VQC\n(spherical)", "FFNN\n(raw Cartesian)"],
                  [vqc_acc, cls_acc], color=["steelblue", "salmon"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy comparison")
    for bar, val in zip(bars, [vqc_acc, cls_acc]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.1%}", ha="center", fontweight="bold")

    plt.tight_layout()
    out_path = FIGURES_DIR / "circles_results.png"
    os.makedirs(out_path.parent, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Figure saved to {out_path}")


# ======================================================================
# 5.  Main
# ======================================================================

def main():
    (X_train, X_test, y_train, y_test,
     Xb_train, Xb_test, yb_train, yb_test,
     X_full, y_full, _) = load_data()

    vqc_res, model, weights, vqc_acc = train_vqc(Xb_train, yb_train, Xb_test, yb_test)
    cls_res, baseline, cls_acc = train_classical(X_train, y_train, X_test, y_test)

    import torch
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({"weights": weights, "accuracy": vqc_acc}, MODELS_DIR / "circles_vqc.pt")
    torch.save(baseline.model.state_dict(), MODELS_DIR / "circles_ffnn.pt")
    print(f"Models saved to {MODELS_DIR}")

    print(f"\n{'='*40}")
    print(f"VQC accuracy:  {vqc_acc:.2%}")
    print(f"FFNN accuracy: {cls_acc:.2%}")
    print(f"{'='*40}")

    plot_results(X_full, y_full, vqc_res, cls_res, vqc_acc, cls_acc)


if __name__ == "__main__":
    main()
