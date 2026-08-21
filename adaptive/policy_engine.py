"""
================================================================================
adaptive/policy_engine.py
================================================================================

Core mathematical decision engine for the Adaptive Threshold Engine (ATE).

Derives continuous, interpretable hyperparameter policies for Generation-(N+1)
recursive learning without hard-coded magic thresholds.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from adaptive.adaptive_config import AdaptiveConfig
from adaptive.scrs_loader import SCRSData
from adaptive.utils import get_adaptive_logger, sigmoid, clamp


@dataclass
class TrainingPolicy:
    """
    Derived continuous training policy specification for Generation-(N+1).

    Attributes
    ----------
    synthetic_ratio : float
        Proportion of synthetic dataset in Generation-(N+1) mix [0, 1].
    anchor_ratio : float
        Proportion of frozen anchor ground-truth dataset in mix [0, 1].
    recommended_epochs : int
        Recommended training epochs for Generation-(N+1).
    recommended_learning_rate : float
        Recommended learning rate for student optimization.
    sampling_temperature : float
        Recommended generation sampling temperature for synthetic generator.
    max_generation_depth : int
        Recommended allowed recursive generation depth.
    continue_recursive_training : bool
        Flag indicating if recursive training should proceed.
    risk_sensitivity_score : float
        Smooth logistic risk factor S in [0, 1].
    """

    synthetic_ratio: float
    anchor_ratio: float
    recommended_epochs: int
    recommended_learning_rate: float
    sampling_temperature: float
    max_generation_depth: int
    continue_recursive_training: bool
    risk_sensitivity_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to JSON-serializable dictionary."""
        return {
            "synthetic_ratio": round(self.synthetic_ratio, 4),
            "anchor_ratio": round(self.anchor_ratio, 4),
            "recommended_epochs": int(self.recommended_epochs),
            "recommended_learning_rate": float(f"{self.recommended_learning_rate:.6e}"),
            "sampling_temperature": round(self.sampling_temperature, 4),
            "max_generation_depth": int(self.max_generation_depth),
            "continue_recursive_training": bool(self.continue_recursive_training),
            "risk_sensitivity_score": round(self.risk_sensitivity_score, 4),
        }


@dataclass
class ATEPolicyResult:
    """
    Complete output container from the Adaptive Threshold Engine.

    Attributes
    ----------
    training_status : str
        Interpretability label (SAFE, MODERATE_RISK, HIGH_RISK, CRITICAL_COLLAPSE).
    policy : TrainingPolicy
        Derived numerical hyperparameter policy.
    metric_influence : Dict[str, float]
        Relative influence of individual SCRS metrics on the policy decision.
    scrs_summary : Dict[str, float]
        Key upstream SCRS scores for reference.
    """

    training_status: str
    policy: TrainingPolicy
    metric_influence: Dict[str, float]
    scrs_summary: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert complete result to dictionary for JSON report."""
        return {
            "training_status": self.training_status,
            "policy": self.policy.to_dict(),
            "scrs_summary": self.scrs_summary,
            "metric_influence": self.metric_influence,
        }


class PolicyEngine:
    """
    Evaluates representation risk, uncertainty risk, overall SCRS, and individual
    metrics to derive continuous, scientifically justified training policies.
    """

    def __init__(
        self,
        config: Optional[AdaptiveConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or AdaptiveConfig()
        self.logger = logger or get_adaptive_logger("adaptive.policy_engine")

    def derive_policy(self, scrs_data: SCRSData) -> ATEPolicyResult:
        """
        Derives the Generation-(N+1) policy from SCRS metrics using smooth continuous
        mathematical functions.

        Parameters
        ----------
        scrs_data : SCRSData
            Parsed SCRS report containing all metrics.

        Returns
        -------
        ATEPolicyResult
            Synthesized policy and decision breakdown.
        """
        scrs = scrs_data.overall_scrs
        rep_risk = scrs_data.representation_risk
        unc_risk = scrs_data.uncertainty_risk

        self.logger.info(
            "Deriving policy for SCRS = %.4f (Rep Risk: %.4f, Unc Risk: %.4f)",
            scrs,
            rep_risk,
            unc_risk,
        )

        # 1. Smooth Logistic Risk Sensitivity Score S in (0, 1)
        k = self.config.sigmoid_steepness
        x0 = self.config.sigmoid_midpoint
        risk_sensitivity = sigmoid(scrs, k=k, x0=x0)

        # 2. Continuous Synthetic vs. Anchor Data Ratio Derivation
        # As risk sensitivity increases, synthetic ratio smoothly decreases while anchor ratio increases
        syn_min = self.config.min_synthetic_ratio
        syn_max = self.config.max_synthetic_ratio
        
        # Convex combination based on smooth risk sensitivity factor
        synthetic_ratio = syn_max - risk_sensitivity * (syn_max - syn_min)
        synthetic_ratio = clamp(synthetic_ratio, syn_min, syn_max)
        anchor_ratio = clamp(1.0 - synthetic_ratio, self.config.min_anchor_ratio, self.config.max_anchor_ratio)
        
        # Ensure exact normalization
        total_ratio = synthetic_ratio + anchor_ratio
        synthetic_ratio = synthetic_ratio / total_ratio
        anchor_ratio = anchor_ratio / total_ratio

        # 3. Continuous Recommended Learning Rate Derivation
        # Exponential smooth decay as risk sensitivity increases to prevent divergence
        lr_base = self.config.base_learning_rate
        lr_min = self.config.min_learning_rate
        lr_max = self.config.max_learning_rate

        # Quadratic smooth interpolation factor: high risk -> lower learning rate
        decay_factor = (1.0 - risk_sensitivity) ** 2
        rec_lr = lr_min + (lr_base - lr_min) * decay_factor
        rec_lr = clamp(rec_lr, lr_min, lr_max)

        # 4. Recommended Epochs Derivation
        # Prevent overfitting on degenerate synthetic data by scaling epochs inversely with drift
        e_base = self.config.base_epochs
        e_min = self.config.min_epochs
        e_max = self.config.max_epochs
        raw_epochs = e_min + (1.0 - risk_sensitivity) * (e_max - e_min)
        rec_epochs = int(round(raw_epochs))
        rec_epochs = int(clamp(rec_epochs, e_min, e_max))

        # 5. Continuous Sampling Temperature Derivation
        # Lower temperature when representation drift or uncertainty is high to sharpen distribution
        t_base = self.config.base_sampling_temperature
        t_min = self.config.min_sampling_temperature
        t_max = self.config.max_sampling_temperature

        # Combine rep_risk and unc_risk for temperature dampening
        weighted_drift = 0.6 * rep_risk + 0.4 * unc_risk
        temp_reduction = 0.5 * weighted_drift
        rec_temp = t_base * (1.0 - temp_reduction)
        rec_temp = clamp(rec_temp, t_min, t_max)

        # 6. Max Generation Depth Derivation
        d_min = self.config.min_generation_depth
        d_max = self.config.max_generation_depth
        raw_depth = d_min + (1.0 - risk_sensitivity) * (d_max - d_min)
        rec_depth = int(round(raw_depth))
        rec_depth = int(clamp(rec_depth, d_min, d_max))

        # 7. Continuous Recursive Continuation Utility Check
        # Smooth utility cutoff: U = 1.0 - sigmoid(SCRS, k=12, x0=0.82)
        utility = 1.0 - sigmoid(scrs, k=12.0, x0=0.82)
        continue_training = utility >= 0.15

        # 8. Training Status Determination (Continuous Interval Mapping)
        training_status = "SAFE"
        for low, high, label in self.config.risk_status_intervals:
            if low <= scrs < high or (high == 1.0 and scrs == 1.0):
                training_status = label
                break

        # 9. Individual Metric Influence Breakdown
        # Calculate metric influence as weighted score contribution
        metric_influence = {}
        for metric, contrib in scrs_data.metric_contributions.items():
            metric_influence[metric] = round(float(contrib), 4)

        # Build policy container
        policy = TrainingPolicy(
            synthetic_ratio=synthetic_ratio,
            anchor_ratio=anchor_ratio,
            recommended_epochs=rec_epochs,
            recommended_learning_rate=rec_lr,
            sampling_temperature=rec_temp,
            max_generation_depth=rec_depth,
            continue_recursive_training=continue_training,
            risk_sensitivity_score=risk_sensitivity,
        )

        scrs_summary = {
            "overall_scrs": round(scrs, 4),
            "representation_risk": round(rep_risk, 4),
            "uncertainty_risk": round(unc_risk, 4),
            "upstream_risk_label": scrs_data.risk_label,
        }

        result = ATEPolicyResult(
            training_status=training_status,
            policy=policy,
            metric_influence=metric_influence,
            scrs_summary=scrs_summary,
        )

        self.logger.info(
            "Policy derived successfully: Status=%s, Synthetic Ratio=%.2f, Anchor Ratio=%.2f, LR=%.2e, Epochs=%d, Continue=%s",
            training_status,
            synthetic_ratio,
            anchor_ratio,
            rec_lr,
            rec_epochs,
            continue_training,
        )

        return result
