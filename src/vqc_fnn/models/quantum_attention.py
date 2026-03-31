import pennylane as qml
from pennylane import numpy as np
import numpy as onp


class QuantumAttention:
    """
    Quantum Attention Block
    -----------------------
    Supports two similarity methods:
        1. Fidelity (exact state overlap, simulator-only)
        2. SWAP-test (true quantum circuit, shot-based estimation)

    Workflow:
    - Encode classical vectors into quantum states
    - Apply VQC
    - Compute similarity (fidelity OR swap-test)
    - Apply softmax to get attention weights
    """

    def __init__(self, n_qubits=4, n_layers=1, shots=1000):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots

        # device for fidelity method (full statevector)
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # device for swap test (needs 2*n_qubits + ancilla)
        self.swap_dev = qml.device(
            "default.qubit", wires=2 * n_qubits + 1, shots=shots
        )

        # Params for VQC
        shape = (n_layers, n_qubits, 3)
        self.params = np.random.normal(0, 0.1, size=shape)

    # ---------------------------------------------------------
    # QUANTUM ENCODING + VQC
    # ---------------------------------------------------------
    def encode_and_vqc(self, x):
        """
        Encode classical vector x → quantum state and apply VQC.
        Returns full statevector.
        """

        @qml.qnode(self.dev)
        def circuit(params, x):
            # Encode classical input
            qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation="Y")

            # Variational block
            qml.StronglyEntanglingLayers(params, wires=range(self.n_qubits))

            return qml.state()

        return circuit(self.params, x)

    # ---------------------------------------------------------
    # FIDELITY SIMILARITY (FAST)
    # ---------------------------------------------------------
    def fidelity(self, state1, state2):
        """
        Fidelity = |<ψ1 | ψ2>|^2
        """
        overlap = np.vdot(state1, state2)
        return np.abs(overlap) ** 2

    # ---------------------------------------------------------
    # SWAP-TEST SIMILARITY
    # ---------------------------------------------------------
    def swap_test(self, q_vec, k_vec):
        """
        Computes similarity using the SWAP-test.

        Returns an estimated fidelity between q and k.
        """

        anc = 0
        q_wires = list(range(1, 1 + self.n_qubits))
        k_wires = list(range(1 + self.n_qubits, 1 + 2 * self.n_qubits))

        @qml.qnode(self.swap_dev)
        def circuit(params, q_vec, k_vec):
            # Prepare ancilla
            qml.Hadamard(wires=anc)

            # Prepare first state (query)
            qml.AngleEmbedding(q_vec, wires=q_wires, rotation="Y")
            qml.StronglyEntanglingLayers(params, wires=q_wires)

            # Prepare second state (key)
            qml.AngleEmbedding(k_vec, wires=k_wires, rotation="Y")
            qml.StronglyEntanglingLayers(params, wires=k_wires)

            # Controlled SWAP between registers
            for qa, kb in zip(q_wires, k_wires):
                qml.CSWAP(wires=[anc, qa, kb])

            # Final hadamard
            qml.Hadamard(wires=anc)

            return qml.sample(wires=anc)

        # Run the circuit
        samples = circuit(self.params, q_vec, k_vec)

        # Probability the ancilla collapsed to |0>
        p0 = np.mean(samples == 0)

        # Fidelity approximation from swap test
        fidelity_est = 2 * p0 - 1

        # Clip numerical noise to [0, 1]
        return max(0.0, min(1.0, fidelity_est))

    # ---------------------------------------------------------
    # SOFTMAX
    # ---------------------------------------------------------
    def softmax(self, scores):
        exp = np.exp(scores - np.max(scores))
        return exp / np.sum(exp)

    # ---------------------------------------------------------
    # MAIN ATTENTION CALL
    # ---------------------------------------------------------
    def compute_attention(self, queries, keys, method="fidelity"):
        """
        method: "fidelity" OR "swap"
        """

        if method not in ["fidelity", "swap"]:
            raise ValueError("method must be 'fidelity' or 'swap'")

        # Precompute states for fidelity method
        if method == "fidelity":
            query_states = [self.encode_and_vqc(q) for q in queries]
            key_states = [self.encode_and_vqc(k) for k in keys]

        attention_matrix = []

        # Compute similarities
        for i, q in enumerate(queries):
            row = []

            for j, k in enumerate(keys):

                if method == "fidelity":
                    score = self.fidelity(query_states[i], key_states[j])

                elif method == "swap":
                    score = self.swap_test(q, k)

                row.append(score)

            # Normalize row using softmax
            attention_matrix.append(
                self.softmax(np.array(row))
            )

        return np.array(attention_matrix)
