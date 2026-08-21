"""
================================================================================
curriculum/policy_loader.py
================================================================================

Policy loader module for the Curriculum Controller.

Loads and validates adaptive_policy.json from the Adaptive Threshold Engine (ATE).
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from curriculum.curriculum_config import CurriculumConfig
from curriculum.utils import get_curriculum_logger


@dataclass
class ATEPolicyData:
    """
    Container for derived ATE policy hyperparameters and risk metrics.

    Attributes
    ----------
    synthetic_ratio : float
        Target proportion of synthetic data in Generation-(N+1) mix.
    anchor_ratio : float
        Target proportion of human anchor data in Generation-(N+1) mix.
    recommended_epochs : int
        Recommended training epochs for downstream student model.
    recommended_learning_rate : float
        Recommended learning rate for downstream optimization.
    sampling_temperature : float
        Recommended sampling temperature used during dataset generation.
    max_generation_depth : int
        Maximum recursive depth allowance.
    continue_recursive_training : bool
        Flag indicating if recursive pipeline should proceed.
    risk_sensitivity_score : float
        Smooth logistic risk score S.
    training_status : str
        Categorical risk label (SAFE, MODERATE_RISK, HIGH_RISK, CRITICAL_COLLAPSE).
    overall_scrs : float
        Fused Synthetic Collapse Risk Score.
    representation_risk : float
        Representation group risk score.
    uncertainty_risk : float
        Uncertainty group risk score.
    curriculum_instructions : Dict[str, Any]
        Actionable instruction directives from ATE.
    raw_policy : Dict[str, Any]
        Complete raw JSON payload.
    """

    synthetic_ratio: float
    anchor_ratio: float
    recommended_epochs: int
    recommended_learning_rate: float
    sampling_temperature: float
    max_generation_depth: int
    continue_recursive_training: bool
    risk_sensitivity_score: float
    training_status: str
    overall_scrs: float
    representation_risk: float
    uncertainty_risk: float
    curriculum_instructions: Dict[str, Any] = field(default_factory=dict)
    raw_policy: Dict[str, Any] = field(default_factory=dict)


class PolicyLoader:
    """Loads and parses adaptive_policy.json from disk."""

    def __init__(
        self,
        config: Optional[CurriculumConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or CurriculumConfig()
        self.logger = logger or get_curriculum_logger("curriculum.policy_loader")

    def load_policy(self, path: Optional[Path] = None) -> ATEPolicyData:
        """
        Loads and parses adaptive_policy.json.

        Parameters
        ----------
        path : Optional[Path]
            Custom path to adaptive_policy.json. Uses config path if None.

        Returns
        -------
        ATEPolicyData
            Parsed policy object containing all derived hyperparameters.
        """
        target_path = path or self.config.policy_json_path
        self.logger.info("Loading adaptive policy from: %s", target_path)

        if not target_path.exists():
            raise FileNotFoundError(
                f"Adaptive policy JSON not found at: {target_path}. "
                "Ensure adaptive/ module has been executed prior to running curriculum/."
            )

        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        training_status = str(data.get("training_status", "UNKNOWN"))
        policy_dict = data.get("policy", {})
        scrs_dict = data.get("scrs_summary", {})
        recs_dict = data.get("recommendations", {})
        curr_instructions = recs_dict.get("curriculum_instructions", {})

        synthetic_ratio = float(policy_dict.get("synthetic_ratio", 0.70))
        anchor_ratio = float(policy_dict.get("anchor_ratio", 0.30))
        rec_epochs = int(policy_dict.get("recommended_epochs", 3))
        rec_lr = float(policy_dict.get("recommended_learning_rate", 3e-5))
        temp = float(policy_dict.get("sampling_temperature", 0.70))
        depth = int(policy_dict.get("max_generation_depth", 3))
        cont = bool(policy_dict.get("continue_recursive_training", True))
        risk_s = float(policy_dict.get("risk_sensitivity_score", 0.50))

        overall_scrs = float(scrs_dict.get("overall_scrs", 0.50))
        rep_risk = float(scrs_dict.get("representation_risk", 0.50))
        unc_risk = float(scrs_dict.get("uncertainty_risk", 0.50))

        policy_data = ATEPolicyData(
            synthetic_ratio=synthetic_ratio,
            anchor_ratio=anchor_ratio,
            recommended_epochs=rec_epochs,
            recommended_learning_rate=rec_lr,
            sampling_temperature=temp,
            max_generation_depth=depth,
            continue_recursive_training=cont,
            risk_sensitivity_score=risk_s,
            training_status=training_status,
            overall_scrs=overall_scrs,
            representation_risk=rep_risk,
            uncertainty_risk=unc_risk,
            curriculum_instructions=curr_instructions,
            raw_policy=data,
        )

        self.logger.info(
            "Successfully loaded policy: Status=%s, Synthetic Ratio=%.2f, Anchor Ratio=%.2f, LR=%.2e",
            training_status,
            synthetic_ratio,
            anchor_ratio,
            rec_lr,
        )

        return policy_data
