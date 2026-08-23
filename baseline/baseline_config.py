"""
================================================================================
baseline/baseline_config.py
================================================================================

Configuration settings for the Student-2 Baseline Dataset Builder (Control Condition).
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent


@dataclass
class BaselineConfig:
    """
    Configuration settings for the Baseline Dataset Builder.

    Attributes
    ----------
    synthetic_dataset_path : Path
        Path to raw Generation-2 synthetic JSONL file.
    output_dir : Path
        Root output directory for baseline artifacts.
    generation_output_dir : Path
        Target directory for generation_2 baseline files.
    train_jsonl_path : Path
        Path for exported train.jsonl.
    val_jsonl_path : Path
        Path for exported validation.jsonl.
    metadata_json_path : Path
        Path for exported metadata.json.
    summary_txt_path : Path
        Path for exported baseline_summary.txt.
    random_seed : int
        Deterministic random seed for sampling reproducibility.
    train_val_split : float
        Proportion of samples allocated to training set (e.g. 0.90).
    """

    # --- Inputs ---
    synthetic_dataset_path: Path = (
        _PROJECT_ROOT / "data" / "synthetic" / "generation_2" / "generation_2_synthetic.jsonl"
    )

    # --- Output Directories ---
    output_dir: Path = _PROJECT_ROOT / "baseline_out"
    generation_output_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "baseline_out" / "generation_2"
    )

    # --- Output File Paths ---
    train_jsonl_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "baseline_out" / "generation_2" / "train.jsonl"
    )
    val_jsonl_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "baseline_out" / "generation_2" / "validation.jsonl"
    )
    metadata_json_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "baseline_out" / "generation_2" / "metadata.json"
    )
    summary_txt_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "baseline_out" / "generation_2" / "baseline_summary.txt"
    )

    # --- Sampling & Reproducibility ---
    random_seed: int = 42
    train_val_split: float = 0.90

    def __post_init__(self) -> None:
        """Ensure output directories exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generation_output_dir.mkdir(parents=True, exist_ok=True)
