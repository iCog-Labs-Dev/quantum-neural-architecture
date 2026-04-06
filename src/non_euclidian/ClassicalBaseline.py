import numpy as np
import torch
import torch.nn as nn


class FFNN(nn.Module):
    """Simple feed-forward network: input -> hidden -> hidden -> 1 (sigmoid)."""

    def __init__(self, n_input, hidden_size=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ClassicalBaseline:
    """
    Wrapper that mirrors the VQC Trainer interface so experiments
    can compare apples-to-apples.

    Parameters
    ----------
    n_input : int
        Dimensionality of each input sample.
    hidden_size : int
        Width of the hidden layers.
    lr : float
        Learning rate for Adam.
    """

    def __init__(self, n_input, hidden_size=16, lr=0.01):
        
        self.model = FFNN(n_input, hidden_size)
        self.lr = lr

    def param_count(self):
        return sum(p.numel() for p in self.model.parameters())



    def fit(self, X_train, Y_train, epochs=200, X_val=None, Y_val=None, verbose_every=20):
        """
        Train the network and return loss histories.

        Returns
        -------
        dict  {train_history, val_history}
        """
        X_t = torch.tensor(X_train, dtype=torch.float32)
        Y_t = torch.tensor(Y_train, dtype=torch.float32)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        train_history = []
        val_history = []

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            preds = self.model(X_t)
            loss = criterion(preds, Y_t)
            loss.backward()
            optimizer.step()
            train_history.append(loss.item())

            if X_val is not None and Y_val is not None:
                val_loss = self._eval_loss(X_val, Y_val, criterion)
                val_history.append(val_loss)

            if verbose_every and (epoch + 1) % verbose_every == 0:
                msg = f"Epoch {epoch + 1:4d} | Train loss: {loss.item():.5f}"
                if val_history:
                    msg += f" | Val loss: {val_history[-1]:.5f}"
                print(msg)

        return {"train_history": train_history, "val_history": val_history}


    def predict(self, X):
        """Return predicted probabilities as a numpy array."""

        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            return self.model(X_t).numpy()

    def predict_classes(self, X, threshold=0.5):
        """Return hard class predictions (0 or 1)."""

        probs = self.predict(X)
        return (probs >= threshold).astype(int)


    def _eval_loss(self, X, Y, criterion):

        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            Y_t = torch.tensor(Y, dtype=torch.float32)
            preds = self.model(X_t)
            return criterion(preds, Y_t).item()
