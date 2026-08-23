"""
================================================================================
baseline/baseline_validator.py
================================================================================

Data integrity validator module for the Baseline Dataset Builder.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from baseline.baseline_config import BaselineConfig
from baseline.baseline_loader import BaselineRecord
from baseline.utils import get_baseline_logger, compute_text_hash


@dataclass
class ValidationReport:
    """
    Validation summary report container.

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
    synthetic_ratio : float
        Achieved synthetic ratio (1.0 for control baseline).
    anchor_ratio : float
        Achieved anchor ratio (0.0 for control baseline).
    validation_issues : List[str]
        Detailed list of logged warnings/issues.
    """

    is_valid: bool
    total_input_records: int
    total_valid_records: int
    duplicates_removed: int
    empty_removed: int
    corrupted_removed: int
    synthetic_ratio: float
    anchor_ratio: float
    validation_issues: List[str] = field(default_factory=list)


class BaselineValidator:
    """Validates records and cleans baseline dataset collections prior to export."""

    def __init__(
        self,
        config: Optional[BaselineConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or BaselineConfig()
        self.logger = logger or get_baseline_logger("baseline.baseline_validator")

    def validate_and_clean(
        self, records: List[BaselineRecord]
    ) -> Tuple[List[BaselineRecord], ValidationReport]:
        """
        Deduplicates, cleans, and validates dataset records.

        Parameters
        ----------
        records : List[BaselineRecord]
            Input records list.

        Returns
        -------
        Tuple[List[BaselineRecord], ValidationReport]
            Cleaned records list and validation summary report.
        """
        self.logger.info("Validating %d input records for baseline...", len(records))

        seen_hashes = set()
        clean_records: List[BaselineRecord] = []

        duplicates = 0
        empty = 0
        corrupted = 0
        issues = []

        for rec in records:
            # 1. Empty check
            if not rec.text or not rec.text.strip():
                empty += 1
                continue

            # 2. Corrupted / Minimum length check
            if len(rec.text.strip()) < 10:
                corrupted += 1
                continue

            # 3. Duplicate check
            h = compute_text_hash(rec.text)
            if h in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(h)

            # Ensure source is synthetic and provenance is recorded
            rec.source = "synthetic"
            clean_records.append(rec)

        total_clean = len(clean_records)
        syn_count = sum(1 for r in clean_records if r.source == "synthetic")
        anc_count = sum(1 for r in clean_records if r.source == "anchor")

        synthetic_ratio = syn_count / max(1, total_clean)
        anchor_ratio = anc_count / max(1, total_clean)

        if anc_count > 0:
            issues.append(f"CRITICAL ERROR: Human anchor records detected in 100% synthetic baseline ({anc_count} records).")

        if duplicates > 0:
            issues.append(f"Identified and removed {duplicates} duplicate sample records.")
        if empty > 0:
            issues.append(f"Removed {empty} empty records.")
        if corrupted > 0:
            issues.append(f"Removed {corrupted} corrupted/short records.")

        is_valid = total_clean > 0 and anc_count == 0 and synthetic_ratio == 1.0

        report = ValidationReport(
            is_valid=is_valid,
            total_input_records=len(records),
            total_valid_records=total_clean,
            duplicates_removed=duplicates,
            empty_removed=empty,
            corrupted_removed=corrupted,
            synthetic_ratio=synthetic_ratio,
            anchor_ratio=anchor_ratio,
            validation_issues=issues,
        )

        self.logger.info(
            "Baseline validation complete: %d valid records retained (Duplicates: %d, Empty: %d, Corrupted: %d). Synthetic: %.2f, Anchor: %.2f",
            total_clean,
            duplicates,
            empty,
            corrupted,
            synthetic_ratio,
            anchor_ratio,
        )

        return clean_records, report
