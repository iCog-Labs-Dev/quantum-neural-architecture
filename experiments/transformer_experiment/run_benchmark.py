import os
import json
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from QEAM.data.datasets import TextClassificationDataset, get_tokenizer, download_or_load_agnews
from QEAM.models.q_transformer import QTransformer
from QEAM.models.classical_transformer import ClassicalTransformer
from QEAM.training.trainer import Trainer
from QEAM.utils.config import Config, set_seed

set_seed(Config.seed)


SUBSET_SIZE = 200
MAX_SEQ_LEN = 32
BATCH_SIZE = 8
N_QUBITS = 4
N_LAYERS = 1
NUM_BLOCKS = 1
EPOCHS = 3

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)


tokenizer = get_tokenizer(max_seq_len=MAX_SEQ_LEN)
dataset = download_or_load_agnews(max_seq_len=MAX_SEQ_LEN, subset=SUBSET_SIZE)

train_texts = list(dataset["train"]["text"])
train_labels = list(dataset["train"]["label"])
test_texts = list(dataset["test"]["text"])
test_labels = list(dataset["test"]["label"])

num_classes = len(set(train_labels + test_labels))

train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_seq_len=MAX_SEQ_LEN)
test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_seq_len=MAX_SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)




def run_experiment(model_type: str = "quantum"):
    print(f"\n--- Running {model_type.capitalize()} Transformer ---")

    if model_type.lower() == "quantum":
        model = QTransformer(
            vocab_size=Config.vocab_size,
            embed_dim=Config.embed_dim,
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            device=Config.device,
            q_device=Config.q_device,
            num_blocks=NUM_BLOCKS,
            num_classes=num_classes,
            ff_hidden_dim=128,
            max_seq_len=MAX_SEQ_LEN,
            dropout=Config.dropout
        )

    else:
        model = ClassicalTransformer(
            vocab_size=Config.vocab_size,
            embed_dim=Config.embed_dim,
            num_blocks=NUM_BLOCKS,
            num_classes=num_classes,
            ff_hidden_dim=128,
            max_seq_len=MAX_SEQ_LEN,
            dropout=Config.dropout
        )

    model.to(Config.device)

    trainer = Trainer(
        model=model,
        device=Config.device,
        lr=3e-4,
        weight_decay=0.0
    )

    history = trainer.fit(train_loader, test_loader, epochs=EPOCHS, print_every=1)

    metrics = trainer.evaluate(test_loader)

    print(f"{model_type.capitalize()} Transformer | Test Metrics: {metrics}")

    return history, metrics


def plot_histories(hist_q, hist_c):

    epochs = range(1, len(hist_q["train_loss"]) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, hist_q["train_loss"], label="Quantum Train Loss")
    plt.plot(epochs, hist_q["val_loss"], label="Quantum Val Loss")

    plt.plot(epochs, hist_c["train_loss"], linestyle="--", label="Classical Train Loss")
    plt.plot(epochs, hist_c["val_loss"], linestyle="--", label="Classical Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.savefig(f"{RESULTS_DIR}/loss_comparison.png")
    plt.close()

    # Accuracy
    plt.figure()

    plt.plot(epochs, hist_q["val_acc"], label="Quantum Accuracy")
    plt.plot(epochs, hist_c["val_acc"], label="Classical Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Comparison")
    plt.legend()

    plt.savefig(f"{RESULTS_DIR}/accuracy_comparison.png")
    plt.close()

    # F1 Score
    plt.figure()

    plt.plot(epochs, hist_q["val_f1"], label="Quantum F1")
    plt.plot(epochs, hist_c["val_f1"], label="Classical F1")

    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Validation F1 Comparison")
    plt.legend()

    plt.savefig(f"{RESULTS_DIR}/f1_comparison.png")
    plt.close()

    print(f"Plots saved to {RESULTS_DIR}/")



if __name__ == "__main__":

    history_q, metrics_q = run_experiment("quantum")
    history_c, metrics_c = run_experiment("classical")

    results = {
        "quantum": metrics_q,
        "classical": metrics_c
    }

    with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    plot_histories(history_q, history_c)

    print("\n--- Mini Benchmark Summary ---")

    for key, metrics in results.items():
        print(f"{key}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")