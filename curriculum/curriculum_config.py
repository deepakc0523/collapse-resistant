"""
================================================================================
curriculum/curriculum_config.py
================================================================================

Configuration settings and path resolutions for the Curriculum Controller (CC).

Defines dataset paths, output paths, sampling seed parameters, train/validation
splits, and stage scheduling proportions for Generation-2 adaptive dataset creation.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# Resolve project root relative to this file
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent


@dataclass
class CurriculumConfig:
    """
    Configuration settings for the Curriculum Controller.

    Attributes
    ----------
    policy_json_path : Path
        Path to input adaptive_policy.json from ATE.
    anchor_dataset_path : Path
        Path to raw/processed human anchor dataset.
    synthetic_dataset_path : Path
        Path to raw synthetic generation_1 dataset directory.
    output_dir : Path
        Root output directory for curriculum artifacts.
    generation_output_dir : Path
        Target directory for generation_2 dataset files.
    plots_dir : Path
        Target directory for visualization plots.
    train_jsonl_path : Path
        Path for exported train.jsonl.
    val_jsonl_path : Path
        Path for exported validation.jsonl.
    metadata_json_path : Path
        Path for exported metadata.json.
    summary_txt_path : Path
        Path for exported curriculum_summary.txt.
    random_seed : int
        Deterministic random seed for sampling reproducibility.
    total_dataset_size : int
        Target sample count for constructed Generation-2 dataset.
    train_val_split : float
        Proportion of samples allocated to training set (e.g. 0.90).
    stage1_ratio : float
        Fraction of dataset allocated to Stage 1 (Foundation Anchor).
    stage2_ratio : float
        Fraction of dataset allocated to Stage 2 (Transition Blend).
    stage3_ratio : float
        Fraction of dataset allocated to Stage 3 (Target Ratio Exposure).
    """

    # --- Inputs ---
    policy_json_path: Path = _PROJECT_ROOT / "adaptive_out" / "adaptive_policy.json"
    anchor_dataset_path: Path = _PROJECT_ROOT / "data" / "processed" / "clean_wikitext.txt"
    synthetic_dataset_path: Path = _PROJECT_ROOT / "data" / "synthetic" / "generation_1"

    # --- Output Directories ---
    output_dir: Path = _PROJECT_ROOT / "curriculum_out"
    generation_output_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "curriculum_out" / "generation_2"
    )
    plots_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "curriculum_out" / "plots"
    )

    # --- Output File Paths ---
    train_jsonl_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "curriculum_out" / "generation_2" / "train.jsonl"
    )
    val_jsonl_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "curriculum_out" / "generation_2" / "validation.jsonl"
    )
    metadata_json_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "curriculum_out" / "generation_2" / "metadata.json"
    )
    summary_txt_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "curriculum_out" / "generation_2" / "curriculum_summary.txt"
    )

    # --- Sampling & Reproducibility ---
    random_seed: int = 42
    total_dataset_size: int = 1000
    train_val_split: float = 0.90

    # --- 3-Stage Curriculum Schedule Proportions ---
    stage1_ratio: float = 0.25  # 25% Pure Anchor Foundation
    stage2_ratio: float = 0.45  # 45% Transition Interleaved Mix
    stage3_ratio: float = 0.30  # 30% Target Policy Exposure

    def __post_init__(self) -> None:
        """Ensure directories exist and validate schedule bounds."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generation_output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        stage_sum = self.stage1_ratio + self.stage2_ratio + self.stage3_ratio
        if abs(stage_sum - 1.0) > 1e-5:
            raise ValueError(f"Curriculum stage ratios must sum to 1.0, got {stage_sum:.6f}")
