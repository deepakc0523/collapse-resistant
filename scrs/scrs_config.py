"""
================================================================================
scrs/scrs_config.py
================================================================================

Configuration dataclass and settings for the Synthetic Collapse Risk Score (SCRS).

Defines input report locations, output directories, group and metric weights,
normalization bounds, and risk category levels.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, List

# Resolve project root relative to this file
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent


@dataclass
class SCRSConfig:
    """
    Configuration for the Synthetic Collapse Risk Score (SCRS) engine.

    Attributes
    ----------
    probe_report_path : Path
        Path to the upstream Probe representation drift report JSON.
    ensemble_report_path : Path
        Path to the upstream Ensemble variance report JSON.
    output_dir : Path
        Root output directory for SCRS artifacts.
    plots_dir : Path
        Subdirectory for output visualizations.
    report_json_path : Path
        Path for the generated scrs_report.json artifact.
    summary_txt_path : Path
        Path for the generated scrs_summary.txt artifact.
    representation_group_weight : float
        Overall weight assigned to the Representation Risk group (default 0.60).
    uncertainty_group_weight : float
        Overall weight assigned to the Uncertainty Risk group (default 0.40).
    representation_metric_weights : Dict[str, float]
        Weights for metrics within the Representation Risk group.
    uncertainty_metric_weights : Dict[str, float]
        Weights for metrics within the Uncertainty Risk group.
    kl_min : float
        Minimum boundary for KL divergence min-max normalization.
    kl_max : float
        Maximum boundary for KL divergence min-max normalization.
    """

    # --- Upstream Input Paths ---
    probe_report_path: Path = _PROJECT_ROOT / "probe_out" / "representation_drift_report.json"
    ensemble_report_path: Path = _PROJECT_ROOT / "ensemble_out" / "variance_report.json"

    # --- Output Directories ---
    output_dir: Path = _PROJECT_ROOT / "scrs_out"
    plots_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "scrs_out" / "plots")

    # --- Output File Paths ---
    report_json_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "scrs_out" / "scrs_report.json"
    )
    summary_txt_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "scrs_out" / "scrs_summary.txt"
    )

    # --- Group Weights ---
    representation_group_weight: float = 0.60
    uncertainty_group_weight: float = 0.40

    # --- Representation Risk Metric Weights ---
    representation_metric_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "hidden_state_drift": 1.0 / 6.0,
            "embedding_drift": 1.0 / 6.0,
            "attention_drift": 1.0 / 6.0,
            "kl_divergence": 1.0 / 6.0,
            "js_divergence": 1.0 / 6.0,
            "prediction_agreement": 1.0 / 6.0,
        }
    )

    # --- Uncertainty Risk Metric Weights ---
    uncertainty_metric_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "predictive_entropy": 1.0 / 6.0,
            "top1_confidence": 1.0 / 6.0,
            "top5_spread": 1.0 / 6.0,
            "probability_variance": 1.0 / 6.0,
            "confidence_margin": 1.0 / 6.0,
            "mc_dropout_consistency": 1.0 / 6.0,
        }
    )

    # --- Normalization Parameters ---
    kl_min: float = 0.0
    kl_max: float = 10.0

    # --- Risk Level Labels ---
    risk_levels: List[Tuple[float, float, str]] = field(
        default_factory=lambda: [
            (0.00, 0.20, "Very Low"),
            (0.20, 0.40, "Low"),
            (0.40, 0.60, "Moderate"),
            (0.60, 0.80, "High"),
            (0.80, 1.00, "Critical"),
        ]
    )

    def update_paths(
        self,
        output_dir: Optional[Path] = None,
        probe_report_path: Optional[Path] = None,
        ensemble_report_path: Optional[Path] = None,
    ) -> None:
        """Dynamically update input and output paths for custom SCRS evaluation runs."""
        if probe_report_path is not None:
            self.probe_report_path = Path(probe_report_path)
        if ensemble_report_path is not None:
            self.ensemble_report_path = Path(ensemble_report_path)
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.plots_dir = self.output_dir / "plots"
            self.report_json_path = self.output_dir / "scrs_report.json"
            self.summary_txt_path = self.output_dir / "scrs_summary.txt"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            self.report_json_path.parent.mkdir(parents=True, exist_ok=True)
            self.summary_txt_path.parent.mkdir(parents=True, exist_ok=True)

    def __post_init__(self) -> None:
        """Ensure directories exist and validate configurations."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.report_json_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_txt_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate group weight sum
        group_sum = self.representation_group_weight + self.uncertainty_group_weight
        if abs(group_sum - 1.0) > 1e-5:
            raise ValueError(f"Group weights must sum to 1.0, got {group_sum:.6f}")
