"""
================================================================================
curriculum/curriculum_report.py
================================================================================

Summary report generator module for the Curriculum Controller.

Exports human-readable curriculum_summary.txt to curriculum_out/generation_2/.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from curriculum.curriculum_config import CurriculumConfig
from curriculum.policy_loader import ATEPolicyData
from curriculum.curriculum_scheduler import ScheduledCurriculum
from curriculum.dataset_validator import ValidationReport
from curriculum.utils import get_curriculum_logger


class CurriculumReportGenerator:
    """Generates human-readable text summary reports for curriculum construction."""

    def __init__(
        self,
        config: Optional[CurriculumConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or CurriculumConfig()
        self.logger = logger or get_curriculum_logger("curriculum.curriculum_report")

    def generate_report(
        self,
        policy_data: ATEPolicyData,
        scheduled: ScheduledCurriculum,
        validation_report: ValidationReport,
        metadata: Dict[str, Any],
    ) -> Path:
        """
        Generates and writes curriculum_summary.txt to disk.

        Parameters
        ----------
        policy_data : ATEPolicyData
            Policy data object.
        scheduled : ScheduledCurriculum
            Scheduled curriculum object.
        validation_report : ValidationReport
            Validation report object.
        metadata : Dict[str, Any]
            Metadata dictionary payload.

        Returns
        -------
        Path
            Path to exported curriculum_summary.txt.
        """
        out_path = self.config.summary_txt_path
        sizes = metadata.get("dataset_sizes", {})
        comp = scheduled.stage_compositions

        lines = [
            "================================================================================",
            "          CURRICULUM CONTROLLER (CC) GENERATION-2 REPORT                        ",
            "================================================================================",
            f" Timestamp           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f" Target Generation   : Generation-2",
            f" Parent Generation   : Generation-1",
            f" Consumed ATE Policy : {self.config.policy_json_path.name}",
            f" Training Status     : {policy_data.training_status}",
            "--------------------------------------------------------------------------------",
            " DATASET MIXING & SIZE SPECIFICATION                                           ",
            "--------------------------------------------------------------------------------",
            f"  - Total Dataset Size         : {sizes.get('total_samples', 0)} samples",
            f"  - Training Split Count       : {sizes.get('train_samples', 0)} samples ({self.config.train_val_split*100:.0f}%)",
            f"  - Validation Split Count     : {sizes.get('val_samples', 0)} samples ({(1-self.config.train_val_split)*100:.0f}%)",
            f"  - Synthetic Data Mix Ratio   : {policy_data.synthetic_ratio:.4f} ({sizes.get('target_synthetic_count', 0)} samples)",
            f"  - Anchor Data Mix Ratio      : {policy_data.anchor_ratio:.4f} ({sizes.get('target_anchor_count', 0)} samples)",
            f"  - Random Seed                : {self.config.random_seed} (Deterministic)",
            "--------------------------------------------------------------------------------",
            " 3-STAGE PROGRESSIVE CURRICULUM SCHEDULE                                        ",
            "--------------------------------------------------------------------------------",
        ]

        for stage_name, info in comp.items():
            lines.append(
                f"  [{stage_name}] Count: {info.get('count', 0)} | "
                f"Anchor Ratio: {info.get('anchor_ratio', 0.0):.2f} | "
                f"Synthetic Ratio: {info.get('synthetic_ratio', 0.0):.2f}"
            )

        lines.extend([
            "--------------------------------------------------------------------------------",
            " DATASET INTEGRITY & VALIDATION                                                 ",
            "--------------------------------------------------------------------------------",
            f"  - Validation Passed          : {validation_report.is_valid}",
            f"  - Duplicate Samples Removed  : {validation_report.duplicates_removed}",
            f"  - Empty Records Removed      : {validation_report.empty_removed}",
            f"  - Corrupted Records Removed  : {validation_report.corrupted_removed}",
            f"  - Ratio Error Margin         : {validation_report.ratio_error:.6f}",
            "--------------------------------------------------------------------------------",
            " DOWNSTREAM COLAB TRAINING SPECIFICATION                                        ",
            "--------------------------------------------------------------------------------",
            f"  - Recommended Epochs         : {policy_data.recommended_epochs}",
            f"  - Recommended Learning Rate  : {policy_data.recommended_learning_rate:.6e}",
            f"  - Sampling Temperature       : {policy_data.sampling_temperature:.4f}",
            f"  - Target Deployment Directory: {self.config.generation_output_dir}",
            "================================================================================",
        ])

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        self.logger.info("Wrote curriculum summary text report: %s", out_path)
        return out_path
