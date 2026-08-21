"""
================================================================================
adaptive/scrs_loader.py
================================================================================

Loader module for the Synthetic Collapse Risk Score (SCRS) report.

Parses, validates, and exposes all granular representation and uncertainty metrics
from scrs_out/scrs_report.json for downstream adaptive training policy derivation.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from adaptive.adaptive_config import AdaptiveConfig
from adaptive.utils import get_adaptive_logger


@dataclass
class SCRSData:
    """
    Structured container holding all metrics extracted from scrs_report.json.

    Attributes
    ----------
    overall_scrs : float
        Fused Synthetic Collapse Risk Score in [0, 1].
    risk_label : str
        Upstream categorical risk label (e.g. High, Moderate).
    representation_risk : float
        Representation group risk score in [0, 1].
    uncertainty_risk : float
        Uncertainty group risk score in [0, 1].
    rep_metrics : Dict[str, float]
        Normalized metrics for representation risk group.
    unc_metrics : Dict[str, float]
        Normalized metrics for uncertainty risk group.
    metric_contributions : Dict[str, float]
        Individual metric contributions to total SCRS.
    raw_representation_metadata : Dict[str, Any]
        Unnormalized raw representation metrics (cosine similarities, KL, JS, etc.).
    raw_uncertainty_metadata : Dict[str, Any]
        Unnormalized raw uncertainty metrics (entropy, confidence, margin, etc.).
    raw_report : Dict[str, Any]
        Complete raw JSON dict.
    """

    overall_scrs: float
    risk_label: str
    representation_risk: float
    uncertainty_risk: float
    rep_metrics: Dict[str, float]
    unc_metrics: Dict[str, float]
    metric_contributions: Dict[str, float]
    raw_representation_metadata: Dict[str, Any]
    raw_uncertainty_metadata: Dict[str, Any]
    raw_report: Dict[str, Any] = field(default_factory=dict)


class SCRSLoader:
    """Loads and validates the SCRS report JSON artifact."""

    def __init__(
        self,
        config: Optional[AdaptiveConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or AdaptiveConfig()
        self.logger = logger or get_adaptive_logger("adaptive.scrs_loader")

    def load_report(self, path: Optional[Path] = None) -> SCRSData:
        """
        Loads and parses scrs_report.json from disk.

        Parameters
        ----------
        path : Optional[Path]
            Custom path to scrs_report.json. Uses config path if None.

        Returns
        -------
        SCRSData
            Validated structured data object containing all SCRS metrics.
        """
        target_path = path or self.config.scrs_report_path
        self.logger.info("Loading SCRS report from: %s", target_path)

        if not target_path.exists():
            raise FileNotFoundError(
                f"SCRS report not found at path: {target_path}. "
                "Ensure scrs/ module has been executed prior to running adaptive/."
            )

        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validation of required fields
        required_keys = ["overall_scrs", "group_risks", "normalized_metrics"]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Invalid SCRS report format: missing key '{key}'")

        overall_scrs = float(data["overall_scrs"])
        risk_label = str(data.get("risk_label", "Unknown"))
        group_risks = data.get("group_risks", {})
        representation_risk = float(group_risks.get("representation_risk", 0.0))
        uncertainty_risk = float(group_risks.get("uncertainty_risk", 0.0))

        normalized = data.get("normalized_metrics", {})
        rep_metrics = {k: float(v) for k, v in normalized.get("representation_group", {}).items()}
        unc_metrics = {k: float(v) for k, v in normalized.get("uncertainty_group", {}).items()}

        metric_contribs = {
            k: float(v) for k, v in data.get("metric_contributions_to_total_scrs", {}).items()
        }

        norm_meta = data.get("normalization_metadata", {})
        raw_rep_meta = norm_meta.get("representation_metadata", {})
        raw_unc_meta = norm_meta.get("uncertainty_metadata", {})

        scrs_data = SCRSData(
            overall_scrs=overall_scrs,
            risk_label=risk_label,
            representation_risk=representation_risk,
            uncertainty_risk=uncertainty_risk,
            rep_metrics=rep_metrics,
            unc_metrics=unc_metrics,
            metric_contributions=metric_contribs,
            raw_representation_metadata=raw_rep_meta,
            raw_uncertainty_metadata=raw_unc_meta,
            raw_report=data,
        )

        self.logger.info(
            "Successfully parsed SCRS report. SCRS = %.4f (%s) [Rep Risk: %.4f, Unc Risk: %.4f]",
            overall_scrs,
            risk_label,
            representation_risk,
            uncertainty_risk,
        )

        return scrs_data
