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



    def compute_fisher_metrics(self, X, Y):
        """
        Fisher Information metrics following the user's specific extraction steps.
        """
        from utility.generalization_metrics import FisherGeneralizationMetric
        import torch.nn as nn
        
        metric = FisherGeneralizationMetric()
        criterion = nn.BCELoss()
        
        # We need gradients, so use train() or just ensure model is in proper state
        self.model.train() 
        
        X_t = torch.tensor(X, dtype=torch.float32)
        Y_t = torch.tensor(Y, dtype=torch.float32)
        
        for i in range(len(X_t)):
            # 1. Iterate through samples
            x_i = X_t[i:i+1]
            y_i = Y_t[i:i+1]
            
            # 1. Forward pass
            output = self.model(x_i)
            loss = criterion(output, y_i)
            
            # 1. Backprop
            self.model.zero_grad() # 4. Reset gradients (important before backward)
            loss.backward()
            
            # 2. Extract Gradients
            grads = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grads.append(param.grad.view(-1))
            
            if grads:
                # 2. Flatten and concatenate
                g_i = torch.cat(grads)
                # 3. Accumulate outer product
                metric.accumulate(g_i)
                
            # 4. Reset Gradients (CRITICAL after sample)
            self.model.zero_grad()
                
        return metric.compute()


    def fit(self, X_train, Y_train, epochs=200, X_val=None, Y_val=None, verbose_every=20, compute_fisher=False):
        """
        Train the network and return loss histories.

        Returns
        -------
        dict  {train_history, val_history, fisher_history}
        """
        X_t = torch.tensor(X_train, dtype=torch.float32)
        Y_t = torch.tensor(Y_train, dtype=torch.float32)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        train_history = []
        val_history = []
        fisher_history = []

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

            # Fisher metrics
            current_fisher = None
            if compute_fisher and verbose_every and (epoch + 1) % verbose_every == 0:
                current_fisher = self.compute_fisher_metrics(X_train, Y_train)
                fisher_history.append(current_fisher)

            if verbose_every and (epoch + 1) % verbose_every == 0:
                msg = f"Epoch {epoch + 1:4d} | Train loss: {loss.item():.5f}"
                if val_history:
                    msg += f" | Val loss: {val_history[-1]:.5f}"
                
                if current_fisher:
                    ed = current_fisher['effective_dimension']
                    se = current_fisher['spectral_entropy_normalized']
                    gb = current_fisher['generalization_bound']
                    msg += f" | d_eff: {ed:.2f} | Entropy: {se:.3f} | Gen bound: {gb:.4f}"
                
                print(msg)

        return {
            "train_history": train_history, 
            "val_history": val_history,
            "fisher_history": fisher_history
        }


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
