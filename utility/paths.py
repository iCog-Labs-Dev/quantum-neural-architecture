from pathlib import Path


def _find_project_root() -> Path:
    """Walk up from this file until we find setup.py — that's the project root."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "setup.py").exists():
            return parent
    raise RuntimeError("Could not locate project root (no setup.py found)")


PROJECT_ROOT = _find_project_root()
RESULTS_DIR  = PROJECT_ROOT / "results"
FIGURES_DIR  = RESULTS_DIR  / "figures"
MODELS_DIR   = RESULTS_DIR  / "models"
LOGS_DIR     = RESULTS_DIR  / "logs"
