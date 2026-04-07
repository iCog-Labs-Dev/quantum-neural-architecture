import argparse
import os
import sys

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))

if src_path not in sys.path:
    sys.path.insert(0, src_path)

from classical_example import inference as run_classical_inference
from classical_example import train as run_classical_train
from qasnn.run_attention import run_q_attention


def run_pipeline(args: argparse.Namespace) -> None:
    metrics_log = args.metrics_log or os.path.join(os.path.dirname(__file__), "example_log.csv")
    selected_models = ["quantum", "classical"] if args.model == "both" else [args.model]

    for model_type in selected_models:
        print(f"\n=== Running {model_type} attention pipeline in {args.mode} mode ===")
        if model_type == "quantum":
            run_q_attention(
                mode=args.mode,
                dataset=args.dataset,
                model_name=args.quantum_model_name,
                saved_dir=args.saved_dir,
                num_qubits=args.num_qubits,
                num_layers=args.num_layers,
                depth_ebd=args.depth_ebd,
                depth_query=args.depth_query,
                depth_key=args.depth_key,
                depth_value=args.depth_value,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                learning_rate=args.quantum_learning_rate,
                using_validation=args.using_validation,
                text=args.text,
                metrics_log_path=metrics_log,
            )
        else:
            if args.mode == "train":
                run_classical_train(
                    dataset=args.dataset,
                    model_name=args.classical_model_name,
                    saved_dir=args.saved_dir,
                    metrics_log=metrics_log,
                    embedding_dim=args.embedding_dim,
                    hidden_dim=args.hidden_dim,
                    batch_size=args.batch_size,
                    num_epochs=args.num_epochs,
                    learning_rate=args.classical_learning_rate,
                    using_validation=args.using_validation,
                )
            else:
                run_classical_inference(
                    dataset=args.dataset,
                    saved_dir=args.saved_dir,
                    model_name=args.classical_model_name,
                    text=args.text,
                    embedding_dim=args.embedding_dim,
                    hidden_dim=args.hidden_dim,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run quantum and classical attention examples through a single pipeline",
    )
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"])
    parser.add_argument("--model", type=str, default="both", choices=["quantum", "classical", "both"])
    parser.add_argument("--dataset", type=str, required=True, help="Directory containing train.txt, test.txt, and vocab.txt")
    parser.add_argument("--saved_dir", type=str, default="./models/", help="Directory to save trained models")
    parser.add_argument(
        "--metrics_log",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "example_log.csv"),
        help="CSV file used to store metrics for all pipeline runs",
    )
    parser.add_argument("--quantum_model_name", type=str, default="qsann_model", help="Filename for the quantum attention checkpoint")
    parser.add_argument("--classical_model_name", type=str, default="classical_attention", help="Filename for the classical attention checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size used by both models")
    parser.add_argument("--num_epochs", "--epoch", type=int, default=10, help="Number of training epochs for both models")
    parser.add_argument("--using_validation", action="store_true", help="Use validate.txt for model selection")
    parser.add_argument("--text", type=str, default="good movie", help="Text to classify during inference")

    parser.add_argument("--num_qubits", type=int, default=4, help="Number of qubits in the quantum model")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of attention layers in the quantum model")
    parser.add_argument("--depth_ebd", type=int, default=1, help="Depth of the embedding circuit")
    parser.add_argument("--depth_query", type=int, default=1, help="Depth of the query circuit")
    parser.add_argument("--depth_key", type=int, default=1, help="Depth of the key circuit")
    parser.add_argument("--depth_value", type=int, default=1, help="Depth of the value circuit")
    parser.add_argument("--quantum_learning_rate", type=float, default=0.01, help="Learning rate for the quantum model")

    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding size for the classical model")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden size for the classical classifier")
    parser.add_argument("--classical_learning_rate", type=float, default=0.001, help="Learning rate for the classical model")

    run_pipeline(parser.parse_args())
