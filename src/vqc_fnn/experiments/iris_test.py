import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =======================
# Imports
# =======================
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import nn, optim
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

from models.hybrid_qnn import HybridQNN


# =======================
# Data loading
# =======================
def get_iris_loaders(batch_size=16, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Load the Iris dataset and return deterministic PyTorch DataLoaders.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    # Dataset split sizes
    val_size = int(len(dataset) * val_ratio)
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - val_size - test_size

    generator = torch.Generator().manual_seed(seed)

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader


# =======================
# Evaluation
# =======================
def evaluate_model(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)

            total_loss += loss.item() * y.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# =======================
# Training
# =======================
def train_hybrid_qnn(
    n_epochs=10,
    batch_size=16,
    learning_rate=0.01,
    seed=42
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_iris_loaders(batch_size=batch_size, seed=seed)

    num_features = train_loader.dataset.dataset[0][0].shape[0]  # Access underlying dataset
    num_classes = 3

    model = HybridQNN(
        num_features=num_features,
        num_classes=num_classes,
        n_vqc_layers=6,  # Reasonable for Iris dataset
        embedding_type="angle",
        gate_type="Y",
        encoding_strategy="reupload",
        device_type="default.qubit"
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate_model(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{n_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    end_time = time.time()
    print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

    test_loss, test_acc = evaluate_model(model, test_loader, device)
    print(f"\nFinal Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    return model, history, n_epochs


# =======================
# Plotting
# =======================
def plot_history(history, n_epochs):
    epochs = range(1, n_epochs + 1)
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =======================
# Main
# =======================
if __name__ == "__main__":
    model, history, n_epochs = train_hybrid_qnn(
        n_epochs=10,
        batch_size=8,
        learning_rate=0.01,
        seed=42
    )
    plot_history(history, n_epochs)
