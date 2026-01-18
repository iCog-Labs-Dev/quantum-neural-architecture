import numpy as np
import torch

from vqc_fnn.models.input_encoder import InputEncoder
from vqc_fnn.models.vqc_layer import VQCLayer

from vqc_fnn.experiments.expressibility.expressibility import (
    sample_quantum_states,
    compute_fidelities,
    compute_expressibility_kl,
)

from vqc_fnn.experiments.expressibility.plot import (
    plot_fidelity_histogram,
    plot_expressibility_saturation,
)


def run():
    torch.manual_seed(0)
    np.random.seed(0)

    n_qubits = 4
    num_samples = 120
    depths = [1, 2, 3, 4, 5]

    encoder = InputEncoder(
        n_qubits=n_qubits,
        embedding_type="angle",
        rotation="Y",
    )

    expressibilities = []

    for depth in depths:
        vqc = VQCLayer(
            n_qubits=n_qubits,
            encoder=encoder,
            n_layers=depth,
            encoding_strategy="reupload",
        )

        states = sample_quantum_states(vqc, num_samples)
        fidelities = compute_fidelities(states)

        expr = compute_expressibility_kl(
            fidelities,
            n_qubits,
        )
        expressibilities.append(expr)

        plot_fidelity_histogram(
            fidelities,
            n_qubits,
            save_path=f"fidelity_depth_{depth}.png",
        )

    plot_expressibility_saturation(
        depths,
        expressibilities,
        save_path="expressibility_saturation.png",
    )


if __name__ == "__main__":
    run()
