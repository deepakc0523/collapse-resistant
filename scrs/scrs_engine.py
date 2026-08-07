"""
================================================================================
scrs/scrs_engine.py
================================================================================

Core mathematical fusion engine for the Synthetic Collapse Risk Score (SCRS).

Combines upstream metrics from Probe and Ensemble loaders, normalizes them,
applies the weighting engine, determines the descriptive risk level label, and
computes overall metric contributions.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

from scrs.scrs_config import SCRSConfig
from scrs.utils import get_scrs_logger, timed_action
from scrs.probe_loader import ProbeLoader, ProbeMetrics
from scrs.ensemble_loader import EnsembleLoader, EnsembleMetrics
from scrs.normalization import Normalizer, NormalizedRepresentationMetrics, NormalizedUncertaintyMetrics
from scrs.weighting_engine import WeightingEngine


@dataclass
class SCRSResult:
    """Dataclass holding complete results of SCRS fusion computation."""
    scrs: float
    risk_label: str
    representation_risk: float
    uncertainty_risk: float
    representation_metrics: Dict[str, float]
    uncertainty_metrics: Dict[str, float]
    representation_weights: Dict[str, float]
    uncertainty_weights: Dict[str, float]
    group_weights: Dict[str, float]
    metric_contributions: Dict[str, float]
    normalization_metadata: Dict[str, Any]
    probe_metrics: ProbeMetrics
    ensemble_metrics: EnsembleMetrics

    def to_dict(self) -> Dict[str, Any]:
        """Converts result into a serializable dictionary."""
        return {
            "overall_scrs": self.scrs,
            "risk_label": self.risk_label,
            "group_risks": {
                "representation_risk": self.representation_risk,
                "uncertainty_risk": self.uncertainty_risk,
            },
            "configured_weights": {
                "group_weights": self.group_weights,
                "representation_metric_weights": self.representation_weights,
                "uncertainty_metric_weights": self.uncertainty_weights,
            },
            "normalized_metrics": {
                "representation_group": self.representation_metrics,
                "uncertainty_group": self.uncertainty_metrics,
            },
            "metric_contributions_to_total_scrs": self.metric_contributions,
            "normalization_metadata": self.normalization_metadata,
        }


class SCRSEngine:
    """Main execution engine for computing SCRS."""

    def __init__(self, config: Optional[SCRSConfig] = None, logger: Optional[logging.Logger] = None) -> None:
        self.config = config or SCRSConfig()
        self.logger = logger or get_scrs_logger("scrs.scrs_engine")
        self.normalizer = Normalizer(kl_min=self.config.kl_min, kl_max=self.config.kl_max)
        self.weighting_engine = WeightingEngine(self.config, logger=self.logger)

    def get_risk_label(self, scrs: float) -> str:
        """
        Maps a continuous SCRS score in [0.0, 1.0] to a descriptive risk label.

        Levels:
          0.00 - 0.20 -> Very Low
          0.20 - 0.40 -> Low
          0.40 - 0.60 -> Moderate
          0.60 - 0.80 -> High
          0.80 - 1.00 -> Critical
        """
        for low, high, label in self.config.risk_levels:
            if low <= scrs <= high:
                return label
            # Edge case handling for exact 1.0 boundary
            if scrs >= 1.0:
                return "Critical"
        return "Unknown"

    def compute(self) -> SCRSResult:
        """
        Runs the full SCRS mathematical fusion pipeline.

        Returns
        -------
        SCRSResult
            Complete result object containing scores, risks, contributions, and metadata.
        """
        self.logger.info("Initializing SCRS Computation Pipeline...")

        # 1. Load upstream reports
        probe_loader = ProbeLoader(self.config.probe_report_path, logger=self.logger)
        ensemble_loader = EnsembleLoader(self.config.ensemble_report_path, logger=self.logger)

        probe_metrics = probe_loader.load()
        ensemble_metrics = ensemble_loader.load()

        # 2. Normalize metrics
        norm_rep = self.normalizer.normalize_representation(
            hidden_cosine=probe_metrics.hidden_state_cosine_similarity,
            emb_cosine=probe_metrics.embedding_cosine_similarity,
            att_cosine=probe_metrics.attention_cosine_similarity,
            kl_div=probe_metrics.kl_divergence,
            js_div=probe_metrics.js_divergence,
            pred_agree=probe_metrics.prediction_agreement_top1,
        )

        norm_unc = self.normalizer.normalize_uncertainty(
            pred_entropy=ensemble_metrics.predictive_entropy,
            top1_conf=ensemble_metrics.top1_confidence,
            top5_spread=ensemble_metrics.top5_spread,
            prob_var=ensemble_metrics.probability_variance,
            conf_margin=ensemble_metrics.confidence_margin,
            mc_consistency=ensemble_metrics.mc_dropout_consistency,
        )

        rep_dict = norm_rep.to_dict()
        unc_dict = norm_unc.to_dict()

        # 3. Compute group risks
        rep_risk = self.weighting_engine.compute_representation_risk(rep_dict)
        unc_risk = self.weighting_engine.compute_uncertainty_risk(unc_dict)

        # 4. Compute unified SCRS
        overall_scrs = self.weighting_engine.compute_scrs(rep_risk, unc_risk)
        risk_label = self.get_risk_label(overall_scrs)

        # 5. Compute individual metric contributions to overall SCRS
        metric_contributions: Dict[str, float] = {}
        
        # Representation metrics contribution: W_rep * w_i * risk_i
        w_rep = self.weighting_engine.rep_group_weight
        for k, v in rep_dict.items():
            metric_w = self.weighting_engine.rep_metric_weights[k]
            metric_contributions[f"rep_{k}"] = w_rep * metric_w * v

        # Uncertainty metrics contribution: W_unc * w_j * risk_j
        w_unc = self.weighting_engine.unc_group_weight
        for k, v in unc_dict.items():
            metric_w = self.weighting_engine.unc_metric_weights[k]
            metric_contributions[f"unc_{k}"] = w_unc * metric_w * v

        # Combine metadata
        norm_metadata = {
            "representation_metadata": norm_rep.normalization_metadata,
            "uncertainty_metadata": norm_unc.normalization_metadata,
        }

        group_weights = {
            "representation_group": w_rep,
            "uncertainty_group": w_unc,
        }

        self.logger.info(
            "SCRS Pipeline Completed -> Representation Risk: %.4f, Uncertainty Risk: %.4f => Overall SCRS: %.4f [%s]",
            rep_risk,
            unc_risk,
            overall_scrs,
            risk_label,
        )

        return SCRSResult(
            scrs=overall_scrs,
            risk_label=risk_label,
            representation_risk=rep_risk,
            uncertainty_risk=unc_risk,
            representation_metrics=rep_dict,
            uncertainty_metrics=unc_dict,
            representation_weights=self.weighting_engine.rep_metric_weights,
            uncertainty_weights=self.weighting_engine.unc_metric_weights,
            group_weights=group_weights,
            metric_contributions=metric_contributions,
            normalization_metadata=norm_metadata,
            probe_metrics=probe_metrics,
            ensemble_metrics=ensemble_metrics,
        )
