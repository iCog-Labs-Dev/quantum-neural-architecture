# 🌌 Quantum Neural Architecture (QNA)

**Quantum Neural Architecture** is a research-grade, hybrid quantum-classical machine learning and optimization framework. Built natively on **PennyLane** and **PyTorch**, this repository focuses on leveraging Variational Quantum Circuits (VQCs), Quantum Transformers, and Quantum Eigensolvers to explore strict quantum advantages in the NISQ (Noisy Intermediate-Scale Quantum) era.

This project was developed following a strategic research pivot from continuous Quantum Hamiltonian Simulation (QHS) toward discrete, gate-based quantum architectures. It yields novel, benchmarked implementations for Natural Language Processing (NLP), Generative Modeling, and Combinatorial Optimization.

---

## 🚀 Core Architectures

This repository contains six distinct quantum architectures, cleanly separated into functional modules within the `src/` directory:

### 1. Quantum Self-Attention & Transformers (`qasnn` & `qeam`)
* **QSANN:** A pure quantum-first prototype mapping tokens directly to quantum rotation angles.
* **QEAM:** A production-grade hybrid quantum-classical transformer. It replaces the classical scaled dot-product attention with a **Quantum Self-Attention Block**, utilizing quantum kernels and SWAP-tests to score query-key pair similarities in a high-dimensional Hilbert space. Benchmarked against classical PyTorch baselines on the SST-2 dataset.

### 2. Quantum Generative Modeling (`born_machine`)
* Implements a **Quantum Circuit Born Machine (QCBM)**.
* Generates discrete probability distributions by leveraging the inherent probabilistic nature of quantum wavefunctions, minimizing Kullback–Leibler (KL) divergence directly from computational-basis measurements via MMD loss.

### 3. Quantum Approximate Optimization Algorithm (`qaoa`)
* Translates discrete combinatorial optimization problems (like Minimum Set Cover) into QUBOs and Ising Hamiltonians.
* Utilizes Trotterized, alternating phase (Cost) and mixing layers to physically bounce the quantum state toward the absolute minimum of an objective function.

### 4. Variational Quantum Eigensolver (`eigensolver`)
* A baseline VQE implementation to calculate the ground-state energies of simulated molecular Hamiltonians. This forms the foundation for tackling quantum chemistry tasks by minimizing the Rayleigh quotient of trial states.

### 5. Geometry-Aware VQCs (`non_euclidean`)
* Explores VQC classifiers on non-Euclidean topologies.
* Utilizes spherical and cyclical ($R_Z$) embeddings to map angular coordinates onto the Bloch sphere. This naturally captures periodicities and topologies that classical Cartesian scalar features fail to represent.

### 6. Quantum Feed-Forward Networks (`vqc_fnn`)
* General-purpose, parameter-shift optimizable quantum neural networks designed for standard classification tasks to benchmark against classical Feed-Forward Networks.

---

## 📁 Repository Structure

The project strictly follows professional Python packaging standards, cleanly separating reusable library code from execution scripts and scientific experiments.

```text
quantum-neural-architecture/
├── src/                      # Core Quantum/Hybrid Modules (Library Code)
│   ├── qaoa/                 # QAOA optimization pipelines
│   ├── qeam/                 # Quantum Attention & Transformers
│   ├── born_machine/         # Generative Born Machine models
│   ├── eigensolver/          # VQE implementations
│   ├── non_euclidean/        # Spherical/Cyclical embeddings
│   ├── qasnn/                # Quantum Attention core logic
│   └── vqc_fnn/              # Quantum Feed-Forward Neural Networks
├── experiments/              # Execution Scripts & Scientific Runs
│   ├── transformer_experiment/ 
│   ├── attention_experiment/ 
│   └── VQC_experiment/
├── tests/                    # Unit tests for quantum models
├── datasets/                 # NLP text data, vocabularies, and raw datasets
├── models/                   # Saved .pt model checkpoints
├── results/                  # Metric logs and evaluation outputs
│   ├── models/               
│   └── figures/              # Convergence plots and generated PNGs
├── utility/                  # Shared generalization and evaluation metrics
├── .gitignore
├── setup.py                  # Local package installation logic
└── requirements.txt          # Project dependencies

## 🛠️ Installation

We strongly recommend using an isolated Python environment (e.g., `venv` or `conda`).

### 1. Clone the repository

```bash
git clone https://github.com/your-username/quantum-neural-architecture.git
cd quantum-neural-architecture
```

---

### 2. Create and activate a virtual environment

#### Using `venv` (Windows)

```bash
python -m venv venv
venv\Scripts\activate
```

#### Using `venv` (Linux/macOS)

```bash
python -m venv venv
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Install the package in editable mode

```bash
pip install -e .
```


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PennyLane](https://img.shields.io/badge/PennyLane-Quantum_ML-purple.svg)](https://pennylane.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Hybrid_ML-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)