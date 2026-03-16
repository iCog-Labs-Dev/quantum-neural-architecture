import numpy as np
import pennylane as qml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pennylane import numpy as pnp

from src.vqc_fnn.VQCModel import VQCModel
from src.vqc_fnn.Embedding import EmbeddingLayer
from src.vqc_fnn.Ansatz import AnsatzLayer
from src.vqc_fnn.Optimizer import Trainer


def main():
    print("=== VQC-FNN Example with Iris Dataset ===")
    
    # 1. Prepare Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Filter for binary classification (classes 0 and 1)
    mask = y < 2
    X = X[mask]
    y = y[mask]
    
    # Map labels to -1.0 and 1.0 for PauliZ measurements
    y = np.where(y == 0, -1.0, 1.0)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    X_train_pnp = pnp.array(X_train, requires_grad=False)
    y_train_pnp = pnp.array(y_train, requires_grad=False)
    X_test_pnp = pnp.array(X_test, requires_grad=False)
    y_test_pnp = pnp.array(y_test, requires_grad=False)
    
    # 2. Configure building blocks
    n_qubits = 4  # Iris has 4 features, angle embedding requires 1 qubit per feature
    embedding = EmbeddingLayer(method="angle", rotation="Y") 
    ansatz = AnsatzLayer(method="strong", n_layers=2)
    
    # 3. Build VQC Model
    model = VQCModel(
        n_qubits=n_qubits,
        embedding=embedding,
        ansatz=ansatz,
        measurement="expval", 
        dropout_rate=0.0
    )
    
    # 4. Train Model
    trainer = Trainer(model, optimizer_type="adam", stepsize=0.1)
    
    print("Starting training for 20 epochs...")
    results = trainer.fit(
        X_train_pnp, y_train_pnp,
        epochs=20,
        X_val=X_test_pnp, Y_val=y_test_pnp,
        verbose_every=5
    )
    
    # 5. Evaluate
    weights = results["weights"]
    model.eval()
    
    preds_raw = np.array([float(model.forward(x, weights)) for x in X_test_pnp])
    preds_class = np.where(preds_raw >= 0, 1.0, -1.0)
    acc = np.mean(preds_class == y_test)
    
    print(f"\nTraining completed! Final Test Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()

