"""
================================================================================
scrs/normalization.py
================================================================================

Normalization engine for SCRS.

Converts all raw metrics from Probe and Ensemble reports into a unified
Risk Scale [0.0, 1.0], where:
  0.0 -> Good / Low Collapse Risk
  1.0 -> Bad / High Collapse Risk
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple


def clamp(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamps a numeric value strictly within [min_val, max_val]."""
    return max(min_val, min(max_val, float(val)))


@dataclass
class NormalizedRepresentationMetrics:
    """Normalized risk scores for Representation Risk group."""
    hidden_state_drift_risk: float
    embedding_drift_risk: float
    attention_drift_risk: float
    kl_divergence_risk: float
    js_divergence_risk: float
    prediction_agreement_risk: float
    normalization_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        return {
            "hidden_state_drift": self.hidden_state_drift_risk,
            "embedding_drift": self.embedding_drift_risk,
            "attention_drift": self.attention_drift_risk,
            "kl_divergence": self.kl_divergence_risk,
            "js_divergence": self.js_divergence_risk,
            "prediction_agreement": self.prediction_agreement_risk,
        }


@dataclass
class NormalizedUncertaintyMetrics:
    """Normalized risk scores for Uncertainty Risk group."""
    predictive_entropy_risk: float
    top1_confidence_risk: float
    top5_spread_risk: float
    probability_variance_risk: float
    confidence_margin_risk: float
    mc_dropout_consistency_risk: float
    normalization_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        return {
            "predictive_entropy": self.predictive_entropy_risk,
            "top1_confidence": self.top1_confidence_risk,
            "top5_spread": self.top5_spread_risk,
            "probability_variance": self.probability_variance_risk,
            "confidence_margin": self.confidence_margin_risk,
            "mc_dropout_consistency": self.mc_dropout_consistency_risk,
        }


class Normalizer:
    """Main normalization engine for converting raw metrics to Risk Scale [0, 1]."""

    def __init__(self, kl_min: float = 0.0, kl_max: float = 10.0) -> None:
        self.kl_min = kl_min
        self.kl_max = kl_max

    def normalize_representation(
        self,
        hidden_cosine: float,
        emb_cosine: float,
        att_cosine: float,
        kl_div: float,
        js_div: float,
        pred_agree: float,
    ) -> NormalizedRepresentationMetrics:
        """
        Normalizes Probe representation metrics to Risk Scale [0, 1].

        Formulas:
          - Hidden State Drift Risk = 1.0 - hidden_cosine
          - Embedding Drift Risk    = 1.0 - emb_cosine
          - Attention Drift Risk    = 1.0 - att_cosine
          - KL Divergence Risk      = MinMax(kl_div, kl_min, kl_max)
          - JS Divergence Risk      = js_div (clamped to [0, 1], since JSD <= 1.0 in base 2 or <= ln(2))
          - Prediction Agreement Risk = 1.0 - pred_agree
        """
        hidden_risk = clamp(1.0 - hidden_cosine)
        emb_risk = clamp(1.0 - emb_cosine)
        att_risk = clamp(1.0 - att_cosine)

        # Min-Max KL Divergence Normalization
        kl_range = max(1e-6, self.kl_max - self.kl_min)
        kl_risk = clamp((kl_div - self.kl_min) / kl_range)

        # JS Divergence is naturally bounded in [0, 1] if log base 2 (or <= ln(2) ~ 0.6931 in nat)
        # If raw js_div > 1.0, scale by ln(2)
        js_scaled = js_div / math.log(2.0) if js_div > 1.0 else js_div
        js_risk = clamp(js_scaled)

        pred_agree_risk = clamp(1.0 - pred_agree)

        metadata = {
            "kl_min": self.kl_min,
            "kl_max": self.kl_max,
            "raw_kl_divergence": kl_div,
            "raw_js_divergence": js_div,
            "raw_hidden_cosine": hidden_cosine,
            "raw_embedding_cosine": emb_cosine,
            "raw_attention_cosine": att_cosine,
            "raw_prediction_agreement": pred_agree,
        }

        return NormalizedRepresentationMetrics(
            hidden_state_drift_risk=hidden_risk,
            embedding_drift_risk=emb_risk,
            attention_drift_risk=att_risk,
            kl_divergence_risk=kl_risk,
            js_divergence_risk=js_risk,
            prediction_agreement_risk=pred_agree_risk,
            normalization_metadata=metadata,
        )

    def normalize_uncertainty(
        self,
        pred_entropy: float,
        top1_conf: float,
        top5_spread: float,
        prob_var: float,
        conf_margin: float,
        mc_consistency: float,
    ) -> NormalizedUncertaintyMetrics:
        """
        Normalizes Ensemble uncertainty metrics to Risk Scale [0, 1].

        Formulas:
          - Predictive Entropy Risk     = pred_entropy (already in [0, 1])
          - Top-1 Confidence Risk       = 1.0 - top1_conf
          - Top-5 Spread Risk           = 1.0 - top5_spread
          - Probability Variance Risk   = 1.0 - prob_var
          - Confidence Margin Risk      = 1.0 - conf_margin
          - MC Dropout Consistency Risk = 1.0 - mc_consistency
        """
        entropy_risk = clamp(pred_entropy)
        top1_risk = clamp(1.0 - top1_conf)
        spread_risk = clamp(1.0 - top5_spread)
        var_risk = clamp(1.0 - prob_var)
        margin_risk = clamp(1.0 - conf_margin)
        mc_risk = clamp(1.0 - mc_consistency)

        metadata = {
            "raw_predictive_entropy": pred_entropy,
            "raw_top1_confidence": top1_conf,
            "raw_top5_spread": top5_spread,
            "raw_probability_variance": prob_var,
            "raw_confidence_margin": conf_margin,
            "raw_mc_dropout_consistency": mc_consistency,
        }

        return NormalizedUncertaintyMetrics(
            predictive_entropy_risk=entropy_risk,
            top1_confidence_risk=top1_risk,
            top5_spread_risk=spread_risk,
            probability_variance_risk=var_risk,
            confidence_margin_risk=margin_risk,
            mc_dropout_consistency_risk=mc_risk,
            normalization_metadata=metadata,
        )
