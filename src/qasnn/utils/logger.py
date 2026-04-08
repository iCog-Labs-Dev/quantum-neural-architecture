"""Structured CSV metrics logger for QSANN training runs."""

import csv
import os
from typing import Any, Dict, Optional

_FIELDS = ["epoch", "iteration", "split", "loss", "accuracy",
           "learning_rate", "best_model", "num_samples", "extra"]


def initialize_metrics_logger(path: Optional[str]) -> Optional[str]:
    """Create the CSV file with headers if path is given and file doesn't exist."""
    if not path:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_FIELDS).writeheader()
    return path


def log_metrics(
    path: Optional[str],
    epoch: int,
    iteration: int,
    split: str,
    loss: float,
    accuracy: float,
    learning_rate: float,
    best_model: bool = False,
    num_samples: int = 0,
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one row to the metrics CSV."""
    if not path:
        return
    row = {
        "epoch": epoch, "iteration": iteration, "split": split,
        "loss": f"{loss:.6f}", "accuracy": f"{accuracy:.6f}",
        "learning_rate": learning_rate, "best_model": int(best_model),
        "num_samples": num_samples, "extra": str(extra_metrics or {}),
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=_FIELDS).writerow(row)
