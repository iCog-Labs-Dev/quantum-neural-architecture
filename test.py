import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from utility.generalization_metrics import FisherGeneralizationMetric


def get_iris_data():
    iris = load_iris()
    X = iris.data
    y = iris.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_test  = torch.tensor(y_test,  dtype=torch.long)

    return X_train, y_train, X_test, y_test


class SimpleNet(nn.Module):
    def __init__(self, input_dim=4, hidden=4, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),  # raw logits (no softmax/LogSoftmax)
        )

    def forward(self, x):
        return self.net(x)


def train():
    X_train, y_train, X_test, y_test = get_iris_data()
    n_train = X_train.shape[0]  # 120
    n_test  = X_test.shape[0]   # 30

    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()  # works with raw logits

    metric = FisherGeneralizationMetric(delta=0.1)

    print("Total trainable weights :", sum(p.numel() for p in model.parameters()))
    print("Train samples           :", n_train)
    print("Test samples            :", n_test)
    print("=" * 60)

    epochs = 30

    for epoch in range(epochs):
        #   Training pass (full batch)
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        train_preds = outputs.argmax(dim=1)
        train_correct = (train_preds == y_train).sum().item()
        train_acc = train_correct / n_train

        loss.backward()
        optimizer.step()

        metric.reset()

        for i in range(n_train):
            x_i = X_train[i:i+1]
            y_i = y_train[i:i+1]

            # Forward + loss with gradients enabled
            output_i = model(x_i)
            sample_loss = criterion(output_i, y_i)

            # Backprop for this sample
            sample_loss.backward()

            # Collect gradient vector (detached)
            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.detach().clone().view(-1))

            if grads:
                grad_vector = torch.cat(grads)
                metric.accumulate(grad_vector)

            # clear gradients after each sample
            model.zero_grad()

        results = metric.compute()

        #   Evaluation
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()
            test_preds = test_outputs.argmax(dim=1)
            test_correct = (test_preds == y_test).sum().item()
            test_acc = test_correct / n_test

        print(f"Epoch {epoch+1:2d} | "
              f"Train loss: {loss.item():.4f}  train_acc: {train_acc:.4f} | "
              f"Test loss: {test_loss:.4f}  test_acc: {test_acc:.4f} | "
              f"Eff. dim: {results.get('effective_dimension', 0.0):.2f} | "
              f"Gen bound: {results.get('generalization_bound', 0.0):.6f} | "
              f"Spec. entropy: {results.get('spectral_entropy_normalized', 0.0):.4f}" )

        print("-" * 140)

    print("Training finished.")


if __name__ == "__main__":
    train()