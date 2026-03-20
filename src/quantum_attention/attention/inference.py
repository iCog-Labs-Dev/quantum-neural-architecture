

r"""
Inference script for the QSANN model.
"""

from typing import List
import torch

from model import QSANN
from dataset import deal_vocab

def inference(
        text: str, model_path: str, vocab_path: str, classes: List[str],
        num_qubits: int, num_layers: int, depth_ebd: int,
        depth_query: int, depth_key: int, depth_value: int
) -> str:
    r"""
    The inference function. Using the trained model to predict new data.

    Args:
        text: The path of the image to be predicted.
        model_path: The path of the model file.
        vocab_path: The path of the vocabulary file.
        classes: The classes of all the labels.
        num_qubits: The number of the qubits which the quantum circuit contains.
        num_layers: The number of the self-attention layers.
        depth_ebd: The depth of the embedding circuit.
        depth_query: The depth of the query circuit.
        depth_key: The depth of the key circuit.
        depth_value: The depth of the value circuit.

    Returns:
        Return the class which the model predicted.
    """
    word2idx = deal_vocab(vocab_path)
    model = QSANN(
        num_qubits=num_qubits, len_vocab=len(word2idx), num_layers=num_layers,
        depth_ebd=depth_ebd, depth_query=depth_query, depth_key=depth_key, depth_value=depth_value,
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    text = [word2idx.get(word, 0) for word in list(text)]
    prediction = model([text])
    prediction = 0 if prediction[0] < 0.5 else 1
    return classes[prediction]
