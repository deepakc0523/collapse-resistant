"""
================================================================================
student/student_config.py
================================================================================

Configuration for the Student Model Training Pipeline.
Mirrors the Anchor hyperparameters but points to the synthetic dataset.
"""

from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_STUDENT_DIR: Path = Path(__file__).resolve().parent
_PROJECT_DIR: Path = _STUDENT_DIR.parent

# Input data (ONLY Synthetic)
SYNTHETIC_DATA_DIR: Path = _PROJECT_DIR / "data" / "synthetic" / "generation_1"

# Output paths
CHECKPOINT_DIR: Path = _PROJECT_DIR / "checkpoints" / "student_model"
LOG_DIR: Path = CHECKPOINT_DIR / "logs"
TRAINING_LOG_FILE: Path = LOG_DIR / "training_log.txt"
TRAINING_HISTORY_FILE: Path = LOG_DIR / "training_history.json"
MODEL_STATISTICS_FILE: Path = LOG_DIR / "model_statistics.json"

# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
MODEL_TYPE: str = "distilgpt2"

# ---------------------------------------------------------------------------
# Training Hyperparameters (Matching Anchor)
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 8
GRAD_ACCUM_STEPS: int = 4
NUM_EPOCHS: int = 1
LEARNING_RATE: float = 5e-5
WEIGHT_DECAY: float = 0.01
WARMUP_RATIO: float = 0.06
MAX_GRAD_NORM: float = 1.0

# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------
VAL_SPLIT_RATIO: float = 0.02
MAX_SAMPLES: Optional[int] = None  # Use all generated synthetic data
NUM_WORKERS: int = 0               # 0 is safe for Windows / Colab mixed environments
