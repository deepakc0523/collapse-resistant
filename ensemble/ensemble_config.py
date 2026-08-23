"""
================================================================================
ensemble/ensemble_config.py
================================================================================

Configuration dataclass for the Ensemble Variance Monitor (EVM).

All paths are resolved relative to the project root (parent of this file's
directory). No thresholds or collapse-risk parameters exist here — this
module only measures and reports uncertainty.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Resolve paths relative to project root
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent


@dataclass
class EnsembleConfig:
    """
    Configuration for the Ensemble Variance Monitor.

    Attributes
    ----------
    student_model_path : Path
        Path to the Best Student model checkpoint directory.
    dataset_source : Path
        Path to the clean wikitext prompt file.
    output_dir : Path
        Root output directory for all EVM artifacts.
    plots_dir : Path
        Subdirectory for generated visualizations.
    report_json_path : Path
        Path for the SCRS-ready JSON variance report.
    summary_txt_path : Path
        Path for the human-readable summary text file.
    device : str
        Compute device — 'auto', 'cuda', 'cpu', or 'mps'.
    batch_size : int
        Number of prompts processed per forward pass.
    max_prompts : int
        Maximum number of prompts to sample from the dataset.
    prompt_min_tokens : int
        Minimum token length for a valid prompt.
    prompt_max_tokens : int
        Maximum token length for a valid prompt.
    random_seed : int
        Seed for deterministic prompt sampling.
    mc_dropout_passes : int
        Number of stochastic forward passes for Monte-Carlo Dropout.
    top_k_confidence : int
        Number of top tokens to consider for spread and variance metrics.
    """

    # --- Model & Data ---
    student_model_path: Path = _PROJECT_ROOT / "checkpoints" / "student_model" / "best"
    dataset_source: Path = _PROJECT_ROOT / "data" / "processed" / "clean_wikitext.txt"

    # --- Output directories ---
    output_dir: Path = _PROJECT_ROOT / "ensemble_out"
    plots_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "ensemble_out" / "plots")

    # --- Report file paths ---
    report_json_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "ensemble_out" / "variance_report.json"
    )
    summary_txt_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "ensemble_out" / "variance_summary.txt"
    )

    # --- Hardware ---
    device: str = "auto"  # 'auto', 'cuda', 'cpu', or 'mps'

    # --- Sampling parameters ---
    batch_size: int = 2
    max_prompts: int = 100
    prompt_min_tokens: int = 32
    prompt_max_tokens: int = 64
    random_seed: int = 42

    # --- Uncertainty parameters ---
    mc_dropout_passes: int = 10       # N stochastic passes for MC Dropout
    top_k_confidence: int = 5         # Top-k tokens used in spread/variance

    def update_paths(
        self,
        output_dir: Optional[Path] = None,
        student_model_path: Optional[Path] = None,
        dataset_source: Optional[Path] = None,
    ) -> None:
        """Dynamically update paths for custom ensemble evaluation runs."""
        if student_model_path is not None:
            self.student_model_path = Path(student_model_path)
        if dataset_source is not None:
            self.dataset_source = Path(dataset_source)
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.plots_dir = self.output_dir / "plots"
            self.report_json_path = self.output_dir / "variance_report.json"
            self.summary_txt_path = self.output_dir / "variance_summary.txt"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            self.report_json_path.parent.mkdir(parents=True, exist_ok=True)
            self.summary_txt_path.parent.mkdir(parents=True, exist_ok=True)

    def __post_init__(self) -> None:
        """Create all required output directories after initialization."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        # Ensure the parent of report paths exists
        self.report_json_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_txt_path.parent.mkdir(parents=True, exist_ok=True)
