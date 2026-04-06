"""
experiment_sphere.py  --  VQC vs FFNN on 3D spherical data
with Fisher-based generalization metrics (no batching).

Metrics computed per epoch via per-sample gradient accumulation:
  - Effective dimension
  - PAC-style generalization bound
  - Spectral entropy (normalized)
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pennylane import numpy as pnp
import pennylane as qml

from Embedding import EmbeddingLayer
from Ansatz import AnsatzLayer
from VQCModel import VQCModel
from Optimizer import Trainer
from ClassicalBaseline import ClassicalBaseline
from data_utils import generate_spherical_dataset, cartesian_to_spherical


# Data

def load_data(n_samples=400, noise=0.05, seed=42):
    X, y = generate_spherical_dataset(n_samples=n_samples, noise=noise, seed=seed)
    X_sphere = cartesian_to_spherical(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        X_sphere, y, test_size=0.25, random_state=seed
    )
    return X_train, X_test, y_train, y_test, Xs_train, Xs_test, ys_train, ys_test, X, y



# VQC  -- 3 qubits x 2 layers x 3 = 18 params


def train_vqc(Xs_train, ys_train, Xs_test, ys_test,
              n_qubits=3, n_layers=2, epochs=10, stepsize=0.05):

    embedding = EmbeddingLayer(method="spherical")
    ansatz    = AnsatzLayer(method="strong", n_layers=n_layers)
    model     = VQCModel(n_qubits=n_qubits, embedding=embedding,
                         ansatz=ansatz, measurement="expval")

    Xs_tr = pnp.array(Xs_train, requires_grad=False)
    Ys_tr = pnp.array(ys_train * 2 - 1, requires_grad=False)
    Xs_te = pnp.array(Xs_test,  requires_grad=False)
    Ys_te = pnp.array(ys_test  * 2 - 1, requires_grad=False)

    print("\n=== VQC Training ({} qubits, {} layers) ===".format(
        n_qubits, n_layers))

    trainer = Trainer(model, optimizer_type="adam", stepsize=stepsize)
    results = trainer.fit(
        Xs_tr, Ys_tr,
        epochs=epochs,
        X_val=Xs_te, Y_val=Ys_te,
        verbose_every=1,
    )

    weights = results["weights"]
    model.eval()
    preds_raw   = np.array([float(model.forward(x, weights)) for x in Xs_te])
    preds_class = (preds_raw > 0.0).astype(int)
    acc = np.mean(preds_class == ys_test.astype(int))
    print("VQC test accuracy: {:.2%}".format(acc))

    return results, acc



class TinyFFNN(nn.Module):
    def __init__(self, n_input=3, hidden=5):
        super().__init__()
        self.fc1 = nn.Linear(n_input, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x)))).squeeze(-1)

#28 params
def train_classical(X_train, y_train, X_test, y_test, epochs=10, lr=0.05):
    net_wrapper = ClassicalBaseline(n_input=X_train.shape[1], hidden_size=3, lr=lr)
    
    
    print("\n=== Classical FFNN ({} params) ===".format(net_wrapper.param_count()))

    results = net_wrapper.fit(
        X_train, y_train,
        epochs=epochs,
        X_val=X_test, Y_val=y_test,
        verbose_every=1,
    )

    preds = net_wrapper.predict_classes(X_test)
    acc = np.mean(preds == y_test.astype(int))
    print("FFNN test accuracy: {:.2%}".format(acc))

    return results, acc



# Plots

def _extract(res, key):
    return [m.get(key, 0.0) for m in res["fisher_history"]]


def plot_results(X, y, vqc_res, cls_res, vqc_acc, cls_acc):
    fig = plt.figure(figsize=(20, 10))
    epochs_x = range(1, len(vqc_res["fisher_history"]) + 1)

    ax = fig.add_subplot(231, projection="3d")
    ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap="bwr", s=10, alpha=0.6)
    ax.set_title("3D Spherical Dataset")

    ax = fig.add_subplot(232)
    ax.plot(vqc_res["train_history"], label="VQC train")
    ax.plot(cls_res["train_history"], label="FFNN train", alpha=0.7)
    ax.plot(vqc_res["val_history"],   label="VQC val",   linestyle="--")
    ax.plot(cls_res["val_history"],   label="FFNN val",  linestyle="--", alpha=0.7)
    ax.set_title("Loss Curves"); ax.set_xlabel("Epoch"); ax.legend()

    ax = fig.add_subplot(233)
    bars = ax.bar(["VQC (27p)", "FFNN (26p)"], [vqc_acc, cls_acc],
                  color=["steelblue", "salmon"])
    ax.set_ylim(0, 1.05); ax.set_title("Test Accuracy")
    for bar, val in zip(bars, [vqc_acc, cls_acc]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                "{:.1%}".format(val), ha="center", fontweight="bold")

    ax = fig.add_subplot(234)
    ax.plot(epochs_x, _extract(vqc_res, "effective_dimension"), label="VQC")
    ax.plot(epochs_x, _extract(cls_res, "effective_dimension"), label="FFNN")
    ax.set_title("Effective Dimension"); ax.set_xlabel("Epoch"); ax.legend()

    ax = fig.add_subplot(235)
    ax.plot(epochs_x, _extract(vqc_res, "generalization_bound"), label="VQC")
    ax.plot(epochs_x, _extract(cls_res, "generalization_bound"), label="FFNN")
    ax.set_title("Gen. Bound (PAC)"); ax.set_xlabel("Epoch"); ax.legend()

    ax = fig.add_subplot(236)
    ax.plot(epochs_x, _extract(vqc_res, "spectral_entropy_normalized"), label="VQC")
    ax.plot(epochs_x, _extract(cls_res, "spectral_entropy_normalized"), label="FFNN")
    ax.set_title("Spectral Entropy (norm.)"); ax.set_xlabel("Epoch"); ax.legend()

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "sphere_results.png")
    plt.savefig(out, dpi=150)
    print("Figure saved to sphere_results.png")


# Main


def main():
    (X_train, X_test, y_train, y_test,
     Xs_train, Xs_test, ys_train, ys_test,
     X_full, y_full) = load_data()

    vqc_res, vqc_acc = train_vqc(Xs_train, ys_train, Xs_test, ys_test)
    cls_res, cls_acc = train_classical(X_train, y_train, X_test, y_test)

    sep = "=" * 40
    print("\n" + sep)
    print("VQC accuracy:  {:.2%}".format(vqc_acc))
    print("FFNN accuracy: {:.2%}".format(cls_acc))
    print(sep)

    plot_results(X_full, y_full, vqc_res, cls_res, vqc_acc, cls_acc)


if __name__ == "__main__":
    main()