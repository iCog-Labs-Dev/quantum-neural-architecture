import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import random
import numpy as np
import json

from ..data.parity_dataset import load_parity_dataset
from ....utility.generalization_metrics import FisherGeneralizationMetric
from ..attention.quantum_self_attention import QuantumSelfAttention
from ..models.classical_transformer import ClassicalSelfAttention



def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class AttentionBenchmarkModel(nn.Module):

    def __init__(self, attention_module, embed_dim, vocab_size, num_classes):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.attention = attention_module

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):

        x = self.embedding(input_ids)

        x, _ = self.attention(x)

        x = x.mean(dim=1)

        logits = self.classifier(x)

        return logits




def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collect_attention_grads(model):

    grads = []

    for p in model.attention.parameters():

        if p.grad is not None:

            grads.append(p.grad.detach().clone().view(-1))

    if len(grads) == 0:
        return None

    return torch.cat(grads)



def train_model(model, loader, epochs=5):

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss()

    model.train()

    losses = []

    for epoch in range(epochs):

        total_loss = 0

        for batch in loader:

            x = batch["input_ids"]
            y = batch["labels"]

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(logits, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        losses.append(total_loss)

        print(f"Epoch {epoch+1} | loss: {total_loss:.4f}")

    return losses


def compute_fisher(model, loader, num_samples=50):

    criterion = nn.CrossEntropyLoss()

    metric = FisherGeneralizationMetric(delta=0.1)

    model.eval()

    processed = 0

    for batch in loader:

        x = batch["input_ids"]
        y = batch["labels"]

        for i in range(x.shape[0]):

            xi = x[i:i+1]
            yi = y[i:i+1]

            logits = model(xi)

            loss = criterion(logits, yi)

            loss.backward()

            grads = collect_attention_grads(model)

            if grads is not None:

                metric.accumulate(grads)

            model.zero_grad()

            processed += 1

            if processed >= num_samples:
                break

        if processed >= num_samples:
            break

    return metric.compute()

def save_results(q_results, c_results):

    results = {
        "quantum": q_results,
        "classical": c_results
    }

    with open("attention_fisher_results.json", "w") as f:

        json.dump(results, f, indent=4)



def main():



    set_seed(42)

    embed_dim = 4
    n_qubits = 3
    n_layers = 1

    seq_len = 16
    batch_size = 1
    epochs = 5

    vocab_size = 2


    q_device = qml.device("default.qubit", wires=n_qubits)


    train_loader, test_loader, num_classes = load_parity_dataset(
        seq_len=seq_len,
        train_size=200,
        test_size=50,
        batch_size=batch_size
    )

    print("\n===== Quantum Attention =====\n")

    q_attention = QuantumSelfAttention(
        n_qubits=n_qubits,
        n_layers=n_layers,
        device=q_device,
        embed_dim=embed_dim,
        max_seq_len=seq_len
    )

    q_model = AttentionBenchmarkModel(
        q_attention,
        embed_dim,
        vocab_size,
        num_classes
    )

    print("Quantum attention parameters:", count_parameters(q_attention))

    train_model(q_model, train_loader, epochs)

    q_results = compute_fisher(q_model, train_loader)

    print("\nQuantum Fisher Metrics")
    print(q_results)

    

    print("\n===== Classical Attention =====\n")

    c_attention = ClassicalSelfAttention(embed_dim=embed_dim)

    c_model = AttentionBenchmarkModel(
        c_attention,
        embed_dim,
        vocab_size,
        num_classes
    )

    print("Classical attention parameters:", count_parameters(c_attention))

    train_model(c_model, train_loader, epochs)

    c_results = compute_fisher(c_model, train_loader)

    print("\nClassical Fisher Metrics")
    print(c_results)

  

    save_results(q_results, c_results)

    print("\nResults saved to attention_fisher_results.json")



if __name__ == "__main__":
    main()