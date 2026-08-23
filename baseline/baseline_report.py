"""
================================================================================
baseline/baseline_report.py
================================================================================

Summary report generator module for the Student-2 Baseline Dataset Builder.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from baseline.baseline_config import BaselineConfig
from baseline.baseline_validator import ValidationReport
from baseline.utils import get_baseline_logger


class BaselineReportGenerator:
    """Generates human-readable text summary reports for baseline construction."""

    def __init__(
        self,
        config: Optional[BaselineConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or BaselineConfig()
        self.logger = logger or get_baseline_logger("baseline.baseline_report")

    def generate_report(
        self,
        validation_report: ValidationReport,
        metadata: Dict[str, Any],
    ) -> Path:
        """
        Generates and writes baseline_summary.txt to disk.

        Parameters
        ----------
        validation_report : ValidationReport
            Validation report object.
        metadata : Dict[str, Any]
            Metadata dictionary payload.

        Returns
        -------
        Path
            Path to exported baseline_summary.txt.
        """
        out_path = self.config.summary_txt_path
        sizes = metadata.get("dataset_sizes", {})

        lines = [
            "================================================================================",
            "          STUDENT-2 CONTROL BASELINE DATASET REPORT                            ",
            "================================================================================",
            f" Timestamp           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f" Experiment ID       : uncontrolled_recursive_baseline",
            f" Target Generation   : Generation-2",
            f" Parent Model        : Generation-1 (Student-1)",
            f" Synthetic Source    : data/synthetic/generation_2/generation_2_synthetic.jsonl",
            "--------------------------------------------------------------------------------",
            " DATASET MIXING & SIZE SPECIFICATION                                           ",
            "--------------------------------------------------------------------------------",
            f"  - Total Dataset Size         : {sizes.get('total_samples', 0)} samples",
            f"  - Training Split Count       : {sizes.get('train_samples', 0)} samples ({self.config.train_val_split*100:.0f}%)",
            f"  - Validation Split Count     : {sizes.get('val_samples', 0)} samples ({(1-self.config.train_val_split)*100:.0f}%)",
            f"  - Synthetic Data Mix Ratio   : 1.0000 ({sizes.get('synthetic_count', 0)} samples)",
            f"  - Anchor Data Mix Ratio      : 0.0000 (0 samples)",
            f"  - Random Seed                : {self.config.random_seed} (Deterministic)",
            "--------------------------------------------------------------------------------",
            " DATASET INTEGRITY & VALIDATION                                                 ",
            "--------------------------------------------------------------------------------",
            f"  - Validation Passed          : {validation_report.is_valid}",
            f"  - Duplicate Samples Removed  : {validation_report.duplicates_removed}",
            f"  - Empty Records Removed      : {validation_report.empty_removed}",
            f"  - Corrupted Records Removed  : {validation_report.corrupted_removed}",
            "--------------------------------------------------------------------------------",
            " EXPERIMENTAL CONTROL CONDITION                                                 ",
            "--------------------------------------------------------------------------------",
            "  - Condition                  : 100% Uncontrolled Recursive Synthetic Data",
            "  - Curriculum Applied         : None (No 3-stage progressive schedule)",
            "  - ATE Policy Applied         : None (No SCRS risk-sensitive mixture)",
            f"  - Target Deployment Directory: {self.config.generation_output_dir}",
            "================================================================================",
        ]

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        self.logger.info("Wrote baseline summary text report: %s", out_path)
        return out_path
