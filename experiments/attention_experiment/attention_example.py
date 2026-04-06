import argparse
import os
import sys

from qasnn.run_attention import run_q_attention
from born_machine.Ansatz import AnsatzLayer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Quantum Self-Attention Neural Network (QSANN)")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'], help="Mode to run: train or inference")
    parser.add_argument('--dataset', type=str, required=True, help="Directory containing train.txt, test.txt, and vocab.txt")
    parser.add_argument('--model_name', type=str, default='qsann_model', help="Name for the saved model")
    parser.add_argument('--saved_dir', type=str, default='./models/', help="Directory to save the trained model")
    parser.add_argument(
        '--metrics_log',
        type=str,
        default=os.path.join(os.path.dirname(__file__), 'example_log.csv'),
        help="CSV file used to store training loss, accuracy, and evaluation metrics",
    )
    parser.add_argument('--data', type=str, help="Path to the dataset")
    # Model Hyperparameters
    parser.add_argument('--num_qubits', type=int, default=4, help="Number of qubits")
    parser.add_argument('--num_layers', type=int, default=1, help="Number of self-attention layers")
    parser.add_argument('--depth_ebd', type=int, default=1, help="Depth of embedding circuit")
    parser.add_argument('--depth_query', type=int, default=1, help="Depth of query circuit")
    parser.add_argument('--depth_key', type=int, default=1, help="Depth of key circuit")
    parser.add_argument('--depth_value', type=int, default=1, help="Depth of value circuit")
    
    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--using_validation', action='store_true', help="Use validation dataset (dev.txt)")
    
    # Inference arguments
    parser.add_argument('--text', type=str, default="Good movie", help="Text to classify during inference")
    
    args = parser.parse_args()
    run_q_attention(
        mode=args.mode,
        dataset=args.dataset,
        model_name=args.model_name,
        saved_dir=args.saved_dir,
        num_qubits=args.num_qubits,
        num_layers=args.num_layers,
        depth_ebd=args.depth_ebd,
        depth_query=args.depth_query,
        depth_key=args.depth_key,
        depth_value=args.depth_value,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        using_validation=args.using_validation,
        text=args.text,
        metrics_log_path=args.metrics_log,
    )
