"""
================================================================================
curriculum/metadata_generator.py
================================================================================

Metadata generator module for the Curriculum Controller.

Generates complete, reproducible metadata payload for Generation-2 training.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from curriculum.curriculum_config import CurriculumConfig
from curriculum.policy_loader import ATEPolicyData
from curriculum.curriculum_scheduler import ScheduledCurriculum
from curriculum.dataset_validator import ValidationReport
from curriculum.utils import get_curriculum_logger


class MetadataGenerator:
    """Generates complete reproducibility metadata JSON payload for Generation-(N+1)."""

    def __init__(
        self,
        config: Optional[CurriculumConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or CurriculumConfig()
        self.logger = logger or get_curriculum_logger("curriculum.metadata_generator")

    def generate_metadata(
        self,
        policy_data: ATEPolicyData,
        scheduled: ScheduledCurriculum,
        validation_report: ValidationReport,
        train_count: int,
        val_count: int,
    ) -> Dict[str, Any]:
        """
        Synthesizes metadata payload.

        Parameters
        ----------
        policy_data : ATEPolicyData
            ATE policy metrics and hyperparameters.
        scheduled : ScheduledCurriculum
            Curriculum schedule object.
        validation_report : ValidationReport
            Validation report object.
        train_count : int
            Count of samples in training split.
        val_count : int
            Count of samples in validation split.

        Returns
        -------
        Dict[str, Any]
            Complete metadata dictionary.
        """
        self.logger.info("Generating Generation-2 reproducibility metadata...")

        metadata = {
            "generation_id": "generation_2",
            "generation_parent": "generation_1",
            "creation_time": datetime.now().isoformat(),
            "framework_version": "1.0",
            "hyperparameters": {
                "synthetic_ratio": policy_data.synthetic_ratio,
                "anchor_ratio": policy_data.anchor_ratio,
                "learning_rate": policy_data.recommended_learning_rate,
                "epochs": policy_data.recommended_epochs,
                "sampling_temperature": policy_data.sampling_temperature,
                "max_generation_depth": policy_data.max_generation_depth,
                "training_status": policy_data.training_status,
            },
            "upstream_risk_scores": {
                "SCRS": policy_data.overall_scrs,
                "representation_risk": policy_data.representation_risk,
                "uncertainty_risk": policy_data.uncertainty_risk,
                "risk_sensitivity_score": policy_data.risk_sensitivity_score,
            },
            "dataset_sizes": {
                "total_samples": train_count + val_count,
                "train_samples": train_count,
                "val_samples": val_count,
                "target_synthetic_count": int(round((train_count + val_count) * policy_data.synthetic_ratio)),
                "target_anchor_count": (train_count + val_count) - int(round((train_count + val_count) * policy_data.synthetic_ratio)),
            },
            "random_seed": self.config.random_seed,
            "curriculum_schedule": {
                "stage_boundaries": scheduled.stage_boundaries,
                "stage_compositions": scheduled.stage_compositions,
            },
            "validation_summary": {
                "is_valid": validation_report.is_valid,
                "duplicates_removed": validation_report.duplicates_removed,
                "empty_removed": validation_report.empty_removed,
                "corrupted_removed": validation_report.corrupted_removed,
                "ratio_error": round(validation_report.ratio_error, 6),
            },
        }

        return metadata
