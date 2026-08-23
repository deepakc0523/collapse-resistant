"""
================================================================================
baseline/metadata_generator.py
================================================================================

Metadata generator module for the Student-2 Baseline Dataset Builder.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from baseline.baseline_config import BaselineConfig
from baseline.baseline_validator import ValidationReport
from baseline.utils import get_baseline_logger


class MetadataGenerator:
    """Generates complete reproducibility metadata JSON payload for Student-2 Baseline."""

    def __init__(
        self,
        config: Optional[BaselineConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or BaselineConfig()
        self.logger = logger or get_baseline_logger("baseline.metadata_generator")

    def generate_metadata(
        self,
        validation_report: ValidationReport,
        train_count: int,
        val_count: int,
    ) -> Dict[str, Any]:
        """
        Synthesizes metadata payload.

        Parameters
        ----------
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
        self.logger.info("Generating Generation-2 baseline reproducibility metadata...")

        metadata = {
            "experiment": "uncontrolled_recursive_baseline",
            "generation_id": "generation_2",
            "source_generation": 2,
            "parent_student": "generation_1",
            "synthetic_source": "data/synthetic/generation_2/generation_2_synthetic.jsonl",
            "synthetic_source_record_count": validation_report.total_input_records,
            "synthetic_ratio": 1.0,
            "anchor_ratio": 0.0,
            "creation_time": datetime.now().isoformat(),
            "framework_version": "1.0",
            "random_seed": self.config.random_seed,
            "hyperparameters": {
                "synthetic_ratio": 1.0,
                "anchor_ratio": 0.0,
                "curriculum_applied": False,
                "ate_policy_applied": False,
            },
            "dataset_sizes": {
                "total_samples": train_count + val_count,
                "train_samples": train_count,
                "val_samples": val_count,
                "synthetic_count": train_count + val_count,
                "anchor_count": 0,
            },
            "validation_summary": {
                "is_valid": validation_report.is_valid,
                "duplicates_removed": validation_report.duplicates_removed,
                "empty_removed": validation_report.empty_removed,
                "corrupted_removed": validation_report.corrupted_removed,
            },
        }

        return metadata
