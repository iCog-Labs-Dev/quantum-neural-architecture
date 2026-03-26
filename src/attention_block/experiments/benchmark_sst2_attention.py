import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import random
import numpy as np
import json

from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score

from utility.generalization_metrics import FisherGeneralizationMetric

from ..attention.quantum_self_attention import QuantumSelfAttention
from ..models.classical_transformer import ClassicalSelfAttention




def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# SST-2 Loader (lightweight subset)


def load_sst2_subset(seq_len=16,
                    train_size=200,
                    test_size=50,
                    batch_size=4):

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    dataset = load_dataset("glue", "sst2")

    def tokenize(example):

        return tokenizer(
            example["sentence"],
            padding="max_length",
            truncation=True,
            max_length=seq_len
        )

    dataset = dataset.map(tokenize, batched=True)

    train_dataset = dataset["train"].shuffle(seed=42).select(range(train_size))
    test_dataset = dataset["validation"].shuffle(seed=42).select(range(test_size))

    train_dataset.set_format(type="torch",
                             columns=["input_ids", "label"])

    test_dataset.set_format(type="torch",
                            columns=["input_ids", "label"])

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size)

    return train_loader, test_loader, tokenizer.vocab_size



# Benchmark Model


class AttentionBenchmarkModel(nn.Module):

    def __init__(self,
                 attention_module,
                 embed_dim,
                 vocab_size,
                 num_classes=2):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size,
                                     embed_dim)

        self.attention = attention_module

        self.classifier = nn.Linear(embed_dim,
                                   num_classes)

    def forward(self,
                input_ids):

        x = self.embedding(input_ids)

        x, _ = self.attention(x)

        x = x.mean(dim=1)

        logits = self.classifier(x)

        return logits




def count_parameters(model):

    return sum(p.numel()
               for p in model.parameters()
               if p.requires_grad)


def collect_attention_grads(model):

    grads = []

    for p in model.attention.parameters():

        if p.grad is not None:

            grads.append(
                p.grad.detach().clone().view(-1)
            )

    if len(grads) == 0:

        return None

    return torch.cat(grads)



# Training


def train_model(model,
                loader,
                epochs=4):

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for batch in loader:

            x = batch["input_ids"]
            y = batch["label"]

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(logits,
                             y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} loss:",
              round(total_loss, 4))



# Evaluation Metrics


def evaluate(model,
             loader):

    model.eval()

    preds = []
    labels = []

    criterion = nn.CrossEntropyLoss()

    total_loss = 0

    with torch.no_grad():

        for batch in loader:

            x = batch["input_ids"]
            y = batch["label"]

            logits = model(x)

            loss = criterion(logits,
                             y)

            total_loss += loss.item()

            pred = torch.argmax(logits,
                                dim=1)

            preds.extend(pred.tolist())

            labels.extend(y.tolist())

    acc = accuracy_score(labels,
                         preds)

    f1 = f1_score(labels,
                  preds)

    return acc, f1, total_loss


# Fisher Metrics


def compute_fisher(model,
                   loader,
                   samples=50):

    metric = FisherGeneralizationMetric(
        delta=0.1
    )

    criterion = nn.CrossEntropyLoss()

    model.eval()

    processed = 0

    for batch in loader:

        x = batch["input_ids"]
        y = batch["label"]

        for i in range(x.shape[0]):

            xi = x[i:i+1]
            yi = y[i:i+1]

            model.zero_grad()

            logits = model(xi)

            loss = criterion(logits,
                             yi)

            loss.backward()

            grads = collect_attention_grads(model)

            if grads is not None:

                metric.accumulate(grads)

            processed += 1

            if processed >= samples:

                break

        if processed >= samples:

            break

    return metric.compute()




def save_results(results):

    with open("sst2_attention_benchmark.json",
              "w") as f:

        json.dump(results,
                  f,
                  indent=4)



def main():

    set_seed(42)

    embed_dim = 8
    n_qubits = 5
    n_layers = 1

    seq_len = 16

    train_loader, test_loader, vocab_size = \
        load_sst2_subset(seq_len)

   
    # Quantum Attention
    

    print("\nQuantum Attention\n")

    q_device = qml.device(
        "default.qubit",
        wires=n_qubits
    )

    q_attention = QuantumSelfAttention(
        n_qubits,
        n_layers,
        q_device,
        embed_dim,
        seq_len
    )

    q_model = AttentionBenchmarkModel(
        q_attention,
        embed_dim,
        vocab_size
    )

    print("Parameters:",
          count_parameters(q_attention))

    train_model(q_model,
                train_loader)

    q_acc, q_f1, q_loss = \
        evaluate(q_model,
                 test_loader)

    q_fisher = compute_fisher(
        q_model,
        train_loader
    )

    # Classical Attention
   
    set_seed(42) 
    print("\nClassical Attention\n")

    c_attention = ClassicalSelfAttention(
        embed_dim
    )

    c_model = AttentionBenchmarkModel(
        c_attention,
        embed_dim,
        vocab_size
    )

    print("Parameters:",
          count_parameters(c_attention))

    train_model(c_model,
                train_loader)

    c_acc, c_f1, c_loss = \
        evaluate(c_model,
                 test_loader)

    c_fisher = compute_fisher(
        c_model,
        train_loader
    )

 

    results = {

        "quantum": {

            "accuracy": q_acc,
            "f1": q_f1,
            "loss": q_loss,
            "fisher": q_fisher
        },

        "classical": {

            "accuracy": c_acc,
            "f1": c_f1,
            "loss": c_loss,
            "fisher": c_fisher
        }
    }

    save_results(results)

    print("\nBenchmark complete")
    print(json.dumps(results,
                     indent=2))


if __name__ == "__main__":

    main()