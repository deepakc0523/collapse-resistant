"""
================================================================================
curriculum/dataset_validator.py
================================================================================

Data integrity validator module for the Curriculum Controller.

Validates dataset records for duplicates, empty lines, corrupted items, target mix
ratio compliance, and metadata consistency.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from curriculum.curriculum_config import CurriculumConfig
from curriculum.dataset_loader import DatasetRecord
from curriculum.policy_loader import ATEPolicyData
from curriculum.utils import get_curriculum_logger, compute_text_hash


@dataclass
class ValidationReport:
    """
    Data validation summary report container.

    Attributes
    ----------
    is_valid : bool
        Overall pass/fail status.
    total_input_records : int
        Count of input records evaluated.
    total_valid_records : int
        Count of valid clean records.
    duplicates_removed : int
        Count of duplicate records identified and removed.
    empty_removed : int
        Count of empty records removed.
    corrupted_removed : int
        Count of corrupted/too short records removed.
    target_synthetic_ratio : float
        Target synthetic ratio from ATE policy.
    achieved_synthetic_ratio : float
        Achieved synthetic ratio after validation.
    ratio_error : float
        Absolute discrepancy |target - achieved|.
    validation_issues : List[str]
        Detailed list of logged warnings/issues.
    """

    is_valid: bool
    total_input_records: int
    total_valid_records: int
    duplicates_removed: int
    empty_removed: int
    corrupted_removed: int
    target_synthetic_ratio: float
    achieved_synthetic_ratio: float
    ratio_error: float
    validation_issues: List[str] = field(default_factory=list)


class DatasetValidator:
    """Validates records and cleans dataset collections prior to export."""

    def __init__(
        self,
        config: Optional[CurriculumConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or CurriculumConfig()
        self.logger = logger or get_curriculum_logger("curriculum.dataset_validator")

    def validate_and_clean(
        self, records: List[DatasetRecord], policy_data: ATEPolicyData
    ) -> Tuple[List[DatasetRecord], ValidationReport]:
        """
        Deduplicates, cleans, and validates dataset records.

        Parameters
        ----------
        records : List[DatasetRecord]
            Input records list.
        policy_data : ATEPolicyData
            ATE policy object.

        Returns
        -------
        Tuple[List[DatasetRecord], ValidationReport]
            Cleaned records list and validation summary report.
        """
        self.logger.info("Validating %d input records...", len(records))

        seen_hashes = set()
        clean_records: List[DatasetRecord] = []

        duplicates = 0
        empty = 0
        corrupted = 0
        issues = []

        for rec in records:
            # 1. Empty Check
            if not rec.text or not rec.text.strip():
                empty += 1
                continue

            # 2. Corrupted / Minimum length check
            if len(rec.text.strip()) < 10:
                corrupted += 1
                continue

            # 3. Duplicate Check
            h = compute_text_hash(rec.text)
            if h in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(h)

            clean_records.append(rec)

        total_clean = len(clean_records)
        syn_count = sum(1 for r in clean_records if r.source == "synthetic")
        achieved_syn_ratio = syn_count / max(1, total_clean)
        target_syn_ratio = policy_data.synthetic_ratio
        ratio_err = abs(target_syn_ratio - achieved_syn_ratio)

        if ratio_err > 0.05:
            issues.append(f"Synthetic ratio discrepancy exceeds 5%: Target={target_syn_ratio:.4f}, Achieved={achieved_syn_ratio:.4f}")

        if duplicates > 0:
            issues.append(f"Identified and removed {duplicates} duplicate sample records.")
        if empty > 0:
            issues.append(f"Removed {empty} empty records.")
        if corrupted > 0:
            issues.append(f"Removed {corrupted} corrupted/short records.")

        is_valid = total_clean > 0 and ratio_err <= 0.05

        report = ValidationReport(
            is_valid=is_valid,
            total_input_records=len(records),
            total_valid_records=total_clean,
            duplicates_removed=duplicates,
            empty_removed=empty,
            corrupted_removed=corrupted,
            target_synthetic_ratio=target_syn_ratio,
            achieved_synthetic_ratio=achieved_syn_ratio,
            ratio_error=ratio_err,
            validation_issues=issues,
        )

        self.logger.info(
            "Validation complete: %d valid records retained. Duplicates: %d, Empty: %d, Corrupted: %d",
            total_clean,
            duplicates,
            empty,
            corrupted,
        )

        return clean_records, report
