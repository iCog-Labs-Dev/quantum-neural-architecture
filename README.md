# Quantum Neural Architecture

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Hybrid%20ML-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-Quantum%20ML-6f42c1)](https://pennylane.ai/)
[![Status](https://img.shields.io/badge/status-research%20prototype-0A7)](#)

Quantum Neural Architecture (QNA) is a research-oriented framework for building, benchmarking, and analyzing hybrid quantum-classical learning systems. The project combines PennyLane-based variational quantum circuits with PyTorch training workflows across attention models, generative models, combinatorial optimization, eigensolvers, and geometry-aware classifiers.

The repository is organized as a modular Python codebase: reusable implementations live under `src/`, executable studies live under `experiments/`, and generated artifacts are collected in `results/`.

## Research Scope

QNA explores gate-based quantum machine learning in the NISQ setting, with emphasis on:

- hybrid quantum-classical neural architectures;
- quantum self-attention and transformer-style sequence models;
- variational quantum circuits for supervised classification;
- Quantum Circuit Born Machines for discrete generative modeling;
- QAOA pipelines for QUBO and Ising-form optimization;
- VQE-style energy minimization for quantum chemistry baselines;
- geometry-aware embeddings for periodic and non-Euclidean data.

The code is intended for experimentation, benchmarking, and extension rather than production deployment.

## Architecture Modules

| Module | Purpose |
| --- | --- |
| `src/qasnn` | Quantum self-attention neural network experiments, training, inference, and dataset utilities. |
| `src/QEAM` | Quantum-enhanced attention machinery, including kernels, feature maps, transformer blocks, training utilities, and metrics. |
| `src/born_machine` | Quantum Circuit Born Machine components for learning target probability distributions. |
| `src/QAOA` | Problem formulation, state preparation, ansatz construction, and optimization for QAOA workflows. |
| `src/eigensolver` | VQE-oriented chemistry environment, physics ansatz, baseline models, and energy minimization. |
| `src/non_euclidian` | VQC classifiers and embeddings for spherical, cyclical, and topology-aware feature spaces. |
| `src/vqc_fnn` | General-purpose variational quantum feed-forward network components. |
| `utility` | Shared path helpers, expressibility metrics, and generalization metrics. |

## Repository Layout

```text
quantum-neural-architecture/
|-- datasets/                         # Local text datasets and vocabulary files
|-- experiments/
|   |-- attention_experiment/          # QSANN and classical attention examples
|   |-- transformer_experiment/        # Transformer benchmark scripts
|   `-- VQC_experiment/                # Born machine, QAOA, VQE, and VQC experiments
|-- results/
|   |-- figures/                       # Generated plots and visualizations
|   |-- logs/                          # Training and evaluation logs
|   `-- models/                        # Saved checkpoints and model outputs
|-- src/
|   |-- born_machine/
|   |-- eigensolver/
|   |-- non_euclidian/
|   |-- QAOA/
|   |-- qasnn/
|   |-- QEAM/
|   `-- vqc_fnn/
|-- utility/                           # Shared project utilities
|-- requirements.txt
|-- setup.py
`-- README.md
```

## Installation

Use a fresh Python environment before installing the project dependencies.

```bash
git clone https://github.com/iCog-Labs-Dev/quantum-neural-architecture.git
cd quantum-neural-architecture

python -m venv venv
```

Activate the environment:

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

Install dependencies and the local package:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` already includes `-e .`, but running `pip install -e .` explicitly is useful when developing locally.

## Quick Start

Run a quantum self-attention example:

```bash
python experiments/attention_experiment/attention_example.py --mode train --num_epochs 10
```

Run inference with a trained QSANN checkpoint:

```bash
python experiments/attention_experiment/attention_example.py --mode inference --text "Good movie"
```

Run selected VQC and optimization experiments:

```bash
python experiments/VQC_experiment/qaoa_exp.py
python experiments/VQC_experiment/eigensolver_exp.py
python experiments/VQC_experiment/spherical_exp.py
python experiments/VQC_experiment/cyclical_exp.py
python experiments/VQC_experiment/borm_machine_exp.py
```

Run transformer benchmark scripts after installing the optional Hugging Face NLP dependencies:

```bash
pip install datasets transformers
python experiments/transformer_experiment/benchmark_sst2_attention.py
python experiments/transformer_experiment/run_attention_fisher_benchmark.py
python experiments/transformer_experiment/run_benchmark.py
```

Generated checkpoints, logs, and plots are written to `results/models/`, `results/logs/`, and `results/figures/`.

## Experiment Families

### Quantum Self-Attention

The `qasnn` and `QEAM` modules explore quantum attention mechanisms where token features are encoded into parameterized circuits and compared through quantum kernels or related similarity measures. These experiments are designed to compare quantum-enhanced attention behavior against classical attention baselines.

### Quantum Generative Modeling

The `born_machine` module implements Quantum Circuit Born Machine workflows for learning discrete distributions from measurement samples. The associated experiments track distribution quality and convergence behavior through generated plots and logs.

### Quantum Optimization

The `QAOA` module converts structured combinatorial problems into QUBO and Ising-style formulations, then applies alternating cost and mixer layers optimized through variational training.

### Variational Eigensolving

The `eigensolver` module provides VQE-style components for preparing trial quantum states, estimating energies, and minimizing quantum chemistry objectives.

### Geometry-Aware Classification

The `non_euclidian` and `vqc_fnn` modules support VQC classifiers with embeddings that are better aligned with periodic, spherical, or otherwise non-Cartesian structure.

## Development Notes

- Importable packages are discovered from `src/` through `setup.py`.
- The installed package name is currently `qnn`.
- Several modules use capitalized package directories such as `QAOA` and `QEAM`; match the repository casing when importing on case-sensitive systems.
- The `results/` directory contains generated artifacts that are useful for comparing experiment outputs.
- Some experiment scripts are research prototypes and may require dependency, dataset, or module updates as the codebase evolves.

## Suggested Workflow

1. Install the project in editable mode.
2. Run a small experiment from `experiments/` to verify the environment.
3. Inspect the corresponding module in `src/`.
4. Save new metrics, plots, and checkpoints under `results/`.
5. Add or update focused tests when changing shared model, training, or utility code.

## Project Status

This repository is an active research prototype for quantum neural architecture experiments. Contributions should prioritize reproducibility, clear experiment configuration, documented assumptions, and careful comparison against classical baselines.
