"""
================================================================================
scrs/weighting_engine.py
================================================================================

Weighting engine for SCRS.

Manages configurable group weights (Representation vs. Uncertainty) and
per-metric weights within each group. Ensures all weights sum to 1.0 and
computes weighted mean risk scores.
"""

import logging
from typing import Dict, Any, Optional
from scrs.scrs_config import SCRSConfig
from scrs.utils import get_scrs_logger


class WeightingEngine:
    """Manages weight validation and weighted combination of normalized risks."""

    def __init__(self, config: SCRSConfig, logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.logger = logger or get_scrs_logger("scrs.weighting_engine")
        self._validate_and_normalize_weights()

    def _validate_and_normalize_weights(self) -> None:
        """Validates and normalizes group and metric weight dictionaries."""
        # 1. Group weights
        total_group_weight = (
            self.config.representation_group_weight + self.config.uncertainty_group_weight
        )
        if abs(total_group_weight - 1.0) > 1e-5:
            self.logger.warning(
                "Group weights sum to %.6f (expected 1.0). Re-normalizing.", total_group_weight
            )
            self.rep_group_weight = self.config.representation_group_weight / total_group_weight
            self.unc_group_weight = self.config.uncertainty_group_weight / total_group_weight
        else:
            self.rep_group_weight = self.config.representation_group_weight
            self.unc_group_weight = self.config.uncertainty_group_weight

        # 2. Representation metric weights
        self.rep_metric_weights = self._normalize_dict_weights(
            self.config.representation_metric_weights, "Representation Risk Metrics"
        )

        # 3. Uncertainty metric weights
        self.unc_metric_weights = self._normalize_dict_weights(
            self.config.uncertainty_metric_weights, "Uncertainty Risk Metrics"
        )

    def _normalize_dict_weights(
        self, weights: Dict[str, float], group_name: str
    ) -> Dict[str, float]:
        total = sum(weights.values())
        if total <= 0.0:
            raise ValueError(f"Weight dictionary for {group_name} must sum to a positive number.")
        if abs(total - 1.0) > 1e-5:
            self.logger.warning(
                "Weights for '%s' sum to %.6f (expected 1.0). Re-normalizing.", group_name, total
            )
            return {k: v / total for k, v in weights.items()}
        return dict(weights)

    def compute_representation_risk(self, metric_risks: Dict[str, float]) -> float:
        """
        Computes the weighted mean risk for Representation Risk group.

        Parameters
        ----------
        metric_risks : Dict[str, float]
            Dictionary of normalized metric risks (key -> risk in [0, 1]).

        Returns
        -------
        float
            Weighted representation risk score in [0, 1].
        """
        score = 0.0
        for metric_name, weight in self.rep_metric_weights.items():
            if metric_name not in metric_risks:
                raise KeyError(f"Missing metric '{metric_name}' in representation risk input.")
            score += weight * metric_risks[metric_name]
        return max(0.0, min(1.0, score))

    def compute_uncertainty_risk(self, metric_risks: Dict[str, float]) -> float:
        """
        Computes the weighted mean risk for Uncertainty Risk group.

        Parameters
        ----------
        metric_risks : Dict[str, float]
            Dictionary of normalized metric risks (key -> risk in [0, 1]).

        Returns
        -------
        float
            Weighted uncertainty risk score in [0, 1].
        """
        score = 0.0
        for metric_name, weight in self.unc_metric_weights.items():
            if metric_name not in metric_risks:
                raise KeyError(f"Missing metric '{metric_name}' in uncertainty risk input.")
            score += weight * metric_risks[metric_name]
        return max(0.0, min(1.0, score))

    def compute_scrs(self, rep_risk: float, unc_risk: float) -> float:
        """
        Computes the final Synthetic Collapse Risk Score (SCRS).

        SCRS = Rep_Weight * Rep_Risk + Unc_Weight * Unc_Risk

        Returns
        -------
        float
            Final SCRS in [0.0, 1.0].
        """
        scrs = (self.rep_group_weight * rep_risk) + (self.unc_group_weight * unc_risk)
        return max(0.0, min(1.0, scrs))
