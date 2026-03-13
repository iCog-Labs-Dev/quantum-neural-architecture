import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from ..utils.metrics import accuracy_score, f1_score


class Trainer:
    """
    Args:
        model (nn.Module): Model to train (Quantum or Classical Transformer)
        device (torch.device): Device to run training on (CPU or GPU)
        lr (float): Learning rate for optimizer
        weight_decay (float): Weight decay for optimizer
        criterion (nn.Module, optional): Loss function. Defaults to CrossEntropyLoss.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 lr: float = 3e-5, weight_decay: float = 0.01,
                 criterion: nn.Module = None):

        self.model = model.to(device)
        self.device = device
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """

        Args:
            dataloader (DataLoader): Training data loader

        Returns:
            dict: Dictionary with training loss and accuracy
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in tqdm(dataloader, desc="Training", leave=False):
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)  
            if isinstance(outputs, tuple):  
                logits = outputs[0]
            else:
                logits = outputs

            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        avg_loss = total_loss / len(dataloader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')

        return {"loss": avg_loss, "accuracy": acc, "f1": f1}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict:
        """
        Args:
            dataloader (DataLoader): Evaluation data loader

        Returns:
            dict: Dictionary with loss, accuracy, and F1-score
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            loss = self.criterion(logits, labels)
            total_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        avg_loss = total_loss / len(dataloader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')

        return {"loss": avg_loss, "accuracy": acc, "f1": f1}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = 15, print_every: int = 1):
        """
        Args:
            train_loader (DataLoader): Training dataset loader
            val_loader (DataLoader): Validation dataset loader
            epochs (int): Number of epochs to train
            print_every (int): How often to print metrics

        Returns:
            dict: Training history with metrics per epoch
        """
        history = {"train_loss": [], "train_acc": [], "train_f1": [],
                   "val_loss": [], "val_acc": [], "val_f1": []}

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["train_f1"].append(train_metrics["f1"])

            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["f1"])

            if epoch % print_every == 0:
                print(f"Epoch {epoch}/{epochs} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Train Acc: {train_metrics['accuracy']:.4f} | "
                      f"Train F1: {train_metrics['f1']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Val Acc: {val_metrics['accuracy']:.4f} | "
                      f"Val F1: {val_metrics['f1']:.4f}")

        return history