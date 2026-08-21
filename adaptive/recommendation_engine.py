"""
================================================================================
adaptive/recommendation_engine.py
================================================================================

Scientific recommendation generator for the Adaptive Threshold Engine (ATE).

Synthesizes actionable, interpretable scientific justifications and automated
curriculum directions based on the complete derived policy and metric profile.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from adaptive.adaptive_config import AdaptiveConfig
from adaptive.scrs_loader import SCRSData
from adaptive.policy_engine import ATEPolicyResult
from adaptive.utils import get_adaptive_logger


@dataclass
class RecommendationReport:
    """
    Structured recommendations and scientific justifications for downstream execution.

    Attributes
    ----------
    status_summary : str
        High-level scientific summary of model health and risk profile.
    primary_risk_driver : str
        Identified main driver of collapse risk (Representation vs Uncertainty metric).
    justifications : List[str]
        Detailed mathematical rationale for each policy hyperparameter recommendation.
    curriculum_instructions : Dict[str, Any]
        Direct machine-readable instructions formatted for Curriculum Controller consumption.
    mitigation_actions : List[str]
        Recommended operational mitigation steps if risk is elevated.
    """

    status_summary: str
    primary_risk_driver: str
    justifications: List[str]
    curriculum_instructions: Dict[str, Any]
    mitigation_actions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation report to dictionary."""
        return {
            "status_summary": self.status_summary,
            "primary_risk_driver": self.primary_risk_driver,
            "justifications": self.justifications,
            "curriculum_instructions": self.curriculum_instructions,
            "mitigation_actions": self.mitigation_actions,
        }


class RecommendationEngine:
    """Generates scientifically sound recommendations and justifications for Generation-(N+1)."""

    def __init__(
        self,
        config: Optional[AdaptiveConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or AdaptiveConfig()
        self.logger = logger or get_adaptive_logger("adaptive.recommendation_engine")

    def generate_recommendations(
        self, scrs_data: SCRSData, policy_result: ATEPolicyResult
    ) -> RecommendationReport:
        """
        Builds detailed scientific justifications and instructions for downstream automation.

        Parameters
        ----------
        scrs_data : SCRSData
            Input SCRS metrics.
        policy_result : ATEPolicyResult
            Derived training policy from PolicyEngine.

        Returns
        -------
        RecommendationReport
            Structured recommendation and justification object.
        """
        policy = policy_result.policy
        scrs = scrs_data.overall_scrs
        status = policy_result.training_status

        self.logger.info("Generating scientific recommendations for status: %s", status)

        # 1. Identify Primary Risk Driver
        influences = policy_result.metric_influence
        if influences:
            top_metric = max(influences.items(), key=lambda x: x[1])
            primary_driver = f"{top_metric[0]} (contribution: {top_metric[1]:.4f})"
        else:
            primary_driver = "Uniform baseline distribution"

        # 2. Formulate High-Level Status Summary
        status_summary = (
            f"The model evaluation yields an overall SCRS of {scrs:.4f}, placing Generation-(N) "
            f"in the {status} category. Representation Risk is {scrs_data.representation_risk:.4f} "
            f"and Uncertainty Risk is {scrs_data.uncertainty_risk:.4f}. Primary collapse risk driver "
            f"is identified as {primary_driver}."
        )

        # 3. Formulate Detailed Scientific Justifications
        justifications = []
        justifications.append(
            f"Synthetic vs Anchor Mix Ratio (Synthetic: {policy.synthetic_ratio:.2f}, Anchor: {policy.anchor_ratio:.2f}): "
            f"Derived continuously to balance innovation with canonical anchor regularization. "
            f"Elevated drift in representation requires scaling anchor ratio to {policy.anchor_ratio:.2f} "
            f"to prevent recursive information entropy accumulation."
        )

        justifications.append(
            f"Learning Rate ({policy.recommended_learning_rate:.2e}): "
            f"Adjusted smoothly relative to total risk sensitivity (S={policy.risk_sensitivity_score:.4f}). "
            f"Decaying learning rate stabilizes optimization geometry when hidden state drift is detected."
        )

        justifications.append(
            f"Training Epochs ({policy.recommended_epochs}): "
            f"Scaled down to prevent over-fitting to intermediate synthetic output distributions under drift."
        )

        justifications.append(
            f"Sampling Temperature ({policy.sampling_temperature:.2f}): "
            f"Dampened from baseline {self.config.base_sampling_temperature:.2f} to sharpen probability "
            f"distribution and constrain speculative variance during synthetic data generation."
        )

        justifications.append(
            f"Max Generation Depth ({policy.max_generation_depth}): "
            f"Constrained based on remaining collapse safety margin to prevent deep recursive error propagation."
        )

        # 4. Mitigation Actions
        mitigation_actions = []
        if status == "SAFE":
            mitigation_actions.append("Proceed with standard recursive generation schedule.")
            mitigation_actions.append("Maintain baseline anchor blending (30% anchor, 70% synthetic).")
        elif status == "MODERATE_RISK":
            mitigation_actions.append("Increase anchor data proportion in Generation-(N+1) dataset.")
            mitigation_actions.append("Monitor KL divergence and predictive entropy during next epoch.")
        elif status == "HIGH_RISK":
            mitigation_actions.append("Substantially increase anchor ratio to 50%+ to anchor representations.")
            mitigation_actions.append("Apply constrained learning rate and reduced sampling temperature.")
            mitigation_actions.append("Schedule interim Probe evaluation prior to full Generation-(N+2) rollout.")
        else: # CRITICAL_COLLAPSE
            mitigation_actions.append("Halt pure synthetic recursive expansion immediately.")
            mitigation_actions.append("Initiate anchor-driven recovery training cycle.")
            mitigation_actions.append("Re-evaluate frozen anchor distance vectors before resuming generation.")

        # 5. Formulate Downstream Curriculum Controller Instructions
        curriculum_instructions = {
            "target_generation": "Generation-(N+1)",
            "continue_pipeline": policy.continue_recursive_training,
            "dataset_synthesis_spec": {
                "synthetic_ratio": policy.synthetic_ratio,
                "anchor_ratio": policy.anchor_ratio,
                "sampling_temperature": policy.sampling_temperature,
                "max_depth": policy.max_generation_depth,
            },
            "training_hyperparameters": {
                "epochs": policy.recommended_epochs,
                "learning_rate": policy.recommended_learning_rate,
            },
            "risk_mitigation_mode": status,
            "consumed_scrs_report": self.config.scrs_report_path.name,
        }

        report = RecommendationReport(
            status_summary=status_summary,
            primary_risk_driver=primary_driver,
            justifications=justifications,
            curriculum_instructions=curriculum_instructions,
            mitigation_actions=mitigation_actions,
        )

        self.logger.info("Successfully synthesized recommendation report.")
        return report
