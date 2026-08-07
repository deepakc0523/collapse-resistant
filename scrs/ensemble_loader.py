"""
================================================================================
scrs/ensemble_loader.py
================================================================================

Loader module for Ensemble Variance Monitor reports.

Parses and validates ensemble_out/variance_report.json to extract
the required prediction uncertainty metrics for SCRS computation.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

from scrs.utils import get_scrs_logger


@dataclass
class EnsembleMetrics:
    """Dataclass holding extracted raw Ensemble metrics."""
    predictive_entropy: float
    top1_confidence: float
    top5_spread: float
    probability_variance: float
    confidence_margin: float
    mc_dropout_consistency: float
    raw_data: Dict[str, Any]


class EnsembleLoader:
    """Loads and validates Ensemble report JSON files."""

    def __init__(self, report_path: Path, logger: Optional[logging.Logger] = None) -> None:
        self.report_path = Path(report_path)
        self.logger = logger or get_scrs_logger("scrs.ensemble_loader")

    def load(self) -> EnsembleMetrics:
        """
        Loads the ensemble JSON report and extracts raw uncertainty metrics.

        Returns
        -------
        EnsembleMetrics
            Dataclass containing parsed raw uncertainty metrics.

        Raises
        ------
        FileNotFoundError
            If the JSON report file does not exist.
        KeyError / ValueError
            If expected JSON metrics are missing or malformed.
        """
        if not self.report_path.is_file():
            self.logger.error("Ensemble report not found at %s", self.report_path)
            raise FileNotFoundError(f"Ensemble report file not found: {self.report_path}")

        self.logger.info("Loading Ensemble report from %s", self.report_path)
        with open(self.report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        agg = data.get("aggregate_metrics", {})
        if not agg:
            raise KeyError("Missing 'aggregate_metrics' in Ensemble report.")

        pred_entropy = float(agg.get("mean_predictive_entropy", 0.0))
        top1_conf = float(agg.get("mean_top1_confidence", 0.0))
        top5_spr = float(agg.get("mean_top5_confidence_spread", 0.0))
        prob_var = float(agg.get("mean_probability_variance", 0.0))
        conf_marg = float(agg.get("mean_confidence_margin", 0.0))
        mc_cons = float(agg.get("mean_mc_dropout_consistency", 0.0))

        self.logger.info(
            "Extracted Ensemble Metrics -> Entropy: %.4f, Top1Conf: %.4f, Top5Spread: %.4f, ProbVar: %.4f, ConfMargin: %.4f, MCConsistency: %.4f",
            pred_entropy,
            top1_conf,
            top5_spr,
            prob_var,
            conf_marg,
            mc_cons,
        )

        return EnsembleMetrics(
            predictive_entropy=pred_entropy,
            top1_confidence=top1_conf,
            top5_spread=top5_spr,
            probability_variance=prob_var,
            confidence_margin=conf_marg,
            mc_dropout_consistency=mc_cons,
            raw_data=data,
        )
