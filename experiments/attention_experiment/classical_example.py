import argparse
import os
import sys
from typing import List, Tuple

import torch
import torch.nn as nn

from qasnn.attention.dataset import TextDataset, build_iter, deal_vocab
from qasnn.utils.logger import initialize_metrics_logger, log_metrics

MODEL_TAG = "classical_attention"


class ClassicalTextClassifier(nn.Module):
    """Simple embedding-based text classifier for the QSANN dataset format."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, texts: List[List[int]]) -> torch.Tensor:
        max_len = max(len(tokens) for tokens in texts)
        padded = [tokens + [self.pad_idx] * (max_len - len(tokens)) for tokens in texts]
        inputs = torch.tensor(padded, dtype=torch.long)
        mask = (inputs != self.pad_idx).unsqueeze(-1)
        embeddings = self.embedding(inputs)
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled).squeeze(-1)


def evaluate(model: nn.Module, data_loader: list) -> Tuple[float, float]:
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for texts, labels in data_loader:
            logits = model(texts)
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
            loss = criterion(logits, labels_tensor)
            preds = (torch.sigmoid(logits) >= 0.5).int()

            total_loss += loss.item() * len(labels)
            total_correct += int((preds == labels_tensor.int()).sum().item())
            total_samples += len(labels)

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def train(
    dataset: str,
    model_name: str,
    saved_dir: str,
    metrics_log: str,
    embedding_dim: int,
    hidden_dim: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    using_validation: bool,
) -> None:
    if dataset[-1] != "/":
        dataset += "/"
    os.makedirs(saved_dir, exist_ok=True)

    word2idx = deal_vocab(f"{dataset}vocab.txt")
    train_dataset = TextDataset(file_path=f"{dataset}train.txt", word2idx=word2idx)
    test_dataset = TextDataset(file_path=f"{dataset}test.txt", word2idx=word2idx)
    if using_validation:
        dev_dataset = TextDataset(file_path=f"{dataset}validate.txt", word2idx=word2idx)

    train_iter = build_iter(train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = build_iter(test_dataset, batch_size=batch_size, shuffle=False)
    if using_validation:
        dev_iter = build_iter(dev_dataset, batch_size=batch_size, shuffle=False)

    model = ClassicalTextClassifier(
        vocab_size=len(word2idx),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    model_path = os.path.join(saved_dir, f"{model_name}.pt")

    metrics_log = initialize_metrics_logger(metrics_log)
    best_eval_loss = float("inf")
    total_step = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for texts, labels in train_iter:
            optimizer.zero_grad()
            logits = model(texts)
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
            loss = criterion(logits, labels_tensor)
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(logits) >= 0.5).int()
            epoch_loss += loss.item() * len(labels)
            epoch_correct += int((preds == labels_tensor.int()).sum().item())
            epoch_samples += len(labels)
            total_step += 1

        train_loss = epoch_loss / max(epoch_samples, 1)
        train_acc = epoch_correct / max(epoch_samples, 1)
        log_metrics(
            metrics_log,
            epoch=epoch,
            iteration=total_step,
            split="train",
            loss=train_loss,
            accuracy=train_acc,
            learning_rate=learning_rate,
            num_samples=epoch_samples,
            extra_metrics={"model": MODEL_TAG},
        )

        if using_validation:
            eval_loss, eval_acc = evaluate(model, dev_iter)
            is_best = eval_loss < best_eval_loss
            if is_best:
                best_eval_loss = eval_loss
                torch.save(model.state_dict(), model_path)
            log_metrics(
                metrics_log,
                epoch=epoch,
                iteration=total_step,
                split="validation",
                loss=eval_loss,
                accuracy=eval_acc,
                learning_rate=learning_rate,
                best_model=is_best,
                num_samples=len(dev_dataset),
                extra_metrics={"model": MODEL_TAG},
            )
            print(
                f"Epoch {epoch + 1:3d} | Train loss: {train_loss:.5f}, acc: {train_acc:.2%} | "
                f"Val loss: {eval_loss:.5f}, acc: {eval_acc:.2%}"
            )
        else:
            eval_loss, eval_acc = evaluate(model, test_iter)
            torch.save(model.state_dict(), model_path)
            log_metrics(
                metrics_log,
                epoch=epoch,
                iteration=total_step,
                split="test",
                loss=eval_loss,
                accuracy=eval_acc,
                learning_rate=learning_rate,
                best_model=True,
                num_samples=len(test_dataset),
                extra_metrics={"model": MODEL_TAG},
            )
            print(
                f"Epoch {epoch + 1:3d} | Train loss: {train_loss:.5f}, acc: {train_acc:.2%} | "
                f"Test loss: {eval_loss:.5f}, acc: {eval_acc:.2%}"
            )

    if using_validation and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        torch.save(model.state_dict(), model_path)

    final_test_loss, final_test_acc = evaluate(model, test_iter)
    log_metrics(
        metrics_log,
        epoch=num_epochs - 1,
        iteration=total_step,
        split="final_test",
        loss=final_test_loss,
        accuracy=final_test_acc,
        learning_rate=learning_rate,
        best_model=True,
        num_samples=len(test_dataset),
        extra_metrics={"model": MODEL_TAG},
    )
    print(f"Final test loss: {final_test_loss:.5f}, acc: {final_test_acc:.2%}")


def inference(dataset: str, saved_dir: str, model_name: str, text: str, embedding_dim: int, hidden_dim: int) -> None:
    if dataset[-1] != "/":
        dataset += "/"
    word2idx = deal_vocab(f"{dataset}vocab.txt")
    model = ClassicalTextClassifier(
        vocab_size=len(word2idx),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    )
    model_path = os.path.join(saved_dir, f"{model_name}.pt")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    token_ids = [word2idx.get(word, 0) for word in text.split()]
    with torch.no_grad():
        logits = model([token_ids])
        probability = torch.sigmoid(logits).item()
        prediction = int(probability >= 0.5)

    print(f"Text: {text}")
    print(f"Prediction: {prediction}")
    print(f"Probability: {probability:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classical baseline for the attention dataset")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"])
    parser.add_argument("--dataset", type=str, required=True, help="Directory containing train.txt, test.txt, and vocab.txt")
    parser.add_argument("--model_name", type=str, default=MODEL_TAG, help="Name for the saved model")
    parser.add_argument("--saved_dir", type=str, default="./models/", help="Directory to save the trained model")
    parser.add_argument(
        "--metrics_log",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "example_log.csv"),
        help="CSV file used to store training loss, accuracy, and evaluation metrics",
    )
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding size for token representations")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden size for the classifier")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--using_validation", action="store_true", help="Use validate.txt for model selection")
    parser.add_argument("--text", type=str, default="good movie", help="Text to classify during inference")

    args = parser.parse_args()

    if args.mode == "train":
        train(
            dataset=args.dataset,
            model_name=args.model_name,
            saved_dir=args.saved_dir,
            metrics_log=args.metrics_log,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            using_validation=args.using_validation,
        )
    else:
        inference(
            dataset=args.dataset,
            saved_dir=args.saved_dir,
            model_name=args.model_name,
            text=args.text,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
        )
