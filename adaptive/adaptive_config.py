"""
================================================================================
adaptive/adaptive_config.py
================================================================================

Configuration dataclass and parameters for the Adaptive Threshold Engine (ATE).

Defines path resolutions, hyperparameter bounds, dynamic transfer function parameters,
and risk classification intervals for Generation-(N+1) training policy synthesis.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Any

# Resolve project root relative to this file
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent


@dataclass
class AdaptiveConfig:
    """
    Configuration settings for the Adaptive Threshold Engine (ATE).

    Attributes
    ----------
    scrs_report_path : Path
        Path to input Synthetic Collapse Risk Score (SCRS) JSON report.
    output_dir : Path
        Output directory for policy artifacts.
    plots_dir : Path
        Output directory for visualization plots.
    policy_json_path : Path
        Destination for adaptive_policy.json.
    summary_txt_path : Path
        Destination for adaptive_summary.txt.
    min_synthetic_ratio : float
        Lower bound for recommended synthetic data mix ratio.
    max_synthetic_ratio : float
        Upper bound for recommended synthetic data mix ratio.
    min_anchor_ratio : float
        Lower bound for recommended anchor data mix ratio.
    max_anchor_ratio : float
        Upper bound for recommended anchor data mix ratio.
    base_epochs : int
        Nominal baseline training epochs.
    min_epochs : int
        Minimum recommended training epochs.
    max_epochs : int
        Maximum recommended training epochs.
    base_learning_rate : float
        Nominal baseline learning rate.
    min_learning_rate : float
        Minimum constrained learning rate.
    max_learning_rate : float
        Maximum expanded learning rate.
    base_sampling_temperature : float
        Nominal baseline generation temperature.
    min_sampling_temperature : float
        Minimum generation temperature (to suppress noise under high drift).
    max_sampling_temperature : float
        Maximum generation temperature.
    min_generation_depth : int
        Minimum recursive generation depth.
    max_generation_depth : int
        Maximum allowed recursive generation depth.
    sigmoid_steepness : float
        Steepness coefficient k for logistic transfer functions.
    sigmoid_midpoint : float
        Midpoint shift x0 for logistic transfer functions.
    """

    # --- Input / Output Paths ---
    scrs_report_path: Path = _PROJECT_ROOT / "scrs_out" / "scrs_report.json"
    output_dir: Path = _PROJECT_ROOT / "adaptive_out"
    plots_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "adaptive_out" / "plots"
    )
    policy_json_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "adaptive_out" / "adaptive_policy.json"
    )
    summary_txt_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "adaptive_out" / "adaptive_summary.txt"
    )

    # --- Hyperparameter Interpolation Bounds ---
    min_synthetic_ratio: float = 0.10
    max_synthetic_ratio: float = 0.90
    min_anchor_ratio: float = 0.10
    max_anchor_ratio: float = 0.90

    base_epochs: int = 3
    min_epochs: int = 1
    max_epochs: int = 5

    base_learning_rate: float = 3e-5
    min_learning_rate: float = 1e-6
    max_learning_rate: float = 1e-4

    base_sampling_temperature: float = 0.70
    min_sampling_temperature: float = 0.30
    max_sampling_temperature: float = 1.00

    min_generation_depth: int = 1
    max_generation_depth: int = 5

    # --- Continuous Transfer Function Parameters ---
    sigmoid_steepness: float = 6.0
    sigmoid_midpoint: float = 0.50

    # --- Risk Status Labels & Intervals ---
    risk_status_intervals: List[Tuple[float, float, str]] = field(
        default_factory=lambda: [
            (0.00, 0.25, "SAFE"),
            (0.25, 0.50, "MODERATE_RISK"),
            (0.50, 0.75, "HIGH_RISK"),
            (0.75, 1.00, "CRITICAL_COLLAPSE"),
        ]
    )

    def __post_init__(self) -> None:
        """Ensure destination directories exist and validate config bounds."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.policy_json_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_txt_path.parent.mkdir(parents=True, exist_ok=True)

        if self.min_synthetic_ratio < 0.0 or self.max_synthetic_ratio > 1.0:
            raise ValueError("Synthetic ratio bounds must be within [0.0, 1.0]")
        if self.min_anchor_ratio < 0.0 or self.max_anchor_ratio > 1.0:
            raise ValueError("Anchor ratio bounds must be within [0.0, 1.0]")
