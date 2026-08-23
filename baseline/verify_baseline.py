"""
================================================================================
baseline/verify_baseline.py
================================================================================

Verification suite for the Student-2 Control Baseline Dataset Builder.

Verifies:
  1. Real Gen-2 source exists.
  2. Exactly 1,000 source records are loaded.
  3. No fallback generation occurs (strict fail-loudly check).
  4. No human anchor records are included.
  5. Synthetic proportion = 100%.
  6. Generation provenance is correct (generation=2, parent_student="generation_1").
  7. Duplicate/empty/corrupt record validation works.
  8. Seed reproducibility works.
  9. Train/validation files are created.
  10. Metadata contains required experiment key/values:
      experiment="uncontrolled_recursive_baseline", source_generation=2,
      parent_student="generation_1", synthetic_ratio=1.0, anchor_ratio=0.0.
"""

import sys
import json
import logging
from pathlib import Path

from baseline.baseline_config import BaselineConfig
from baseline.baseline_loader import BaselineLoader, BaselineRecord
from baseline.baseline_validator import BaselineValidator
from baseline.metadata_generator import MetadataGenerator
from baseline.baseline_exporter import BaselineExporter
from baseline.baseline_report import BaselineReportGenerator
from baseline.utils import get_baseline_logger, set_seed


def verify_baseline_module() -> bool:
    """
    Executes all verification checks (1 through 10) for the Baseline Dataset Builder.

    Returns
    -------
    bool
        True if all verification checks pass, False otherwise.
    """
    logger = get_baseline_logger("baseline.verify_baseline")
    logger.info("================================================================================")
    logger.info(" Starting Student-2 Control Baseline Verification Suite (Checks 1-10)...")
    logger.info("================================================================================")

    config = BaselineConfig()
    set_seed(config.random_seed)

    try:
        # --- Check 1 & 2: Real Gen-2 Source exists and loads exactly 1,000 records ---
        logger.info("[1/10 & 2/10] Verifying Real Gen-2 source file existence and 1,000 record count...")
        loader = BaselineLoader(config=config, logger=logger)
        raw_records = loader.load_synthetic_dataset(allow_fallback=False, required_records=1000)

        assert config.synthetic_dataset_path.exists(), f"Source file does not exist: {config.synthetic_dataset_path}"
        assert len(raw_records) == 1000, f"Expected 1,000 records, got {len(raw_records)}"
        logger.info("[PASS] [1] & [2] Real Gen-2 source file exists and loaded exactly 1,000 records.")

        # --- Check 3: Strict fail-loudly behavior (No fallback generation) ---
        logger.info("[3/10] Verifying strict mode fails loudly without fallback data...")
        bad_config = BaselineConfig()
        bad_config.synthetic_dataset_path = Path("non_existent_baseline_gen2.jsonl")
        bad_loader = BaselineLoader(config=bad_config, logger=logger)

        try:
            bad_loader.load_synthetic_dataset(allow_fallback=False)
            assert False, "Expected FileNotFoundError when real synthetic file is missing and allow_fallback=False"
        except FileNotFoundError as e:
            logger.info("[PASS] [3] No fallback data generated: failed loudly as expected (%s).", str(e))

        # --- Check 4 & 5: No human anchor records & 100% synthetic proportion ---
        logger.info("[4/10 & 5/10] Verifying 0 human anchor records and 100% synthetic proportion...")
        validator = BaselineValidator(config=config, logger=logger)
        clean_records, val_report = validator.validate_and_clean(raw_records)

        anchor_count = sum(1 for r in clean_records if r.source == "anchor")
        synthetic_count = sum(1 for r in clean_records if r.source == "synthetic")
        assert anchor_count == 0, f"Expected 0 human anchor records, got {anchor_count}"
        assert synthetic_count == len(clean_records), f"Expected all synthetic records, got {synthetic_count}/{len(clean_records)}"
        assert val_report.synthetic_ratio == 1.0, f"Expected synthetic_ratio 1.0, got {val_report.synthetic_ratio}"
        assert val_report.anchor_ratio == 0.0, f"Expected anchor_ratio 0.0, got {val_report.anchor_ratio}"
        logger.info("[PASS] [4] & [5] Verified 0 human anchor records (100% synthetic ratio).")

        # --- Check 6: Generation Provenance ---
        logger.info("[6/10] Verifying generation provenance metadata...")
        for r in clean_records:
            assert r.metadata.get("generation") == 2, f"Expected generation 2, got {r.metadata.get('generation')}"
            assert r.metadata.get("parent_student") == "generation_1", f"Expected parent_student generation_1, got {r.metadata.get('parent_student')}"
            assert "prompt" in r.metadata, "Record metadata missing prompt"
        logger.info("[PASS] [6] Provenance metadata correctly preserved (generation=2, parent_student=generation_1).")

        # --- Check 7: Duplicate/Empty/Corrupt Record Validation ---
        logger.info("[7/10] Verifying duplicate/empty/corrupt record validation...")
        test_records = list(clean_records)

        # Inject empty record
        test_records.append(BaselineRecord(text="", source="synthetic", record_id="emp_1", metadata={}))
        # Inject corrupt record
        test_records.append(BaselineRecord(text="short", source="synthetic", record_id="cor_1", metadata={}))
        # Inject duplicate record
        test_records.append(clean_records[0])

        _, test_report = validator.validate_and_clean(test_records)
        assert test_report.empty_removed == 1, "Failed to remove empty record"
        assert test_report.corrupted_removed == 1, "Failed to remove corrupted short record"
        assert test_report.duplicates_removed >= 1, "Failed to remove duplicate record"
        logger.info("[PASS] [7] Validation correctly handles empty, corrupted, and duplicate records.")

        # --- Check 8: Seed Reproducibility ---
        logger.info("[8/10] Verifying seed reproducibility...")
        raw2 = loader.load_synthetic_dataset(allow_fallback=False, required_records=1000)
        for r1, r2 in zip(raw_records, raw2):
            assert r1.record_id == r2.record_id, "Record identification must be completely deterministic"
        logger.info("[PASS] [8] Seed reproducibility check passed.")

        # --- Check 9 & 10: File Export & Metadata Key Validation ---
        logger.info("[9/10 & 10/10] Verifying train/validation file creation and metadata identification...")
        train_count = int(round(len(clean_records) * config.train_val_split))
        val_count = len(clean_records) - train_count

        meta_gen = MetadataGenerator(config=config, logger=logger)
        metadata_payload = meta_gen.generate_metadata(
            validation_report=val_report,
            train_count=train_count,
            val_count=val_count,
        )

        exporter = BaselineExporter(config=config, logger=logger)
        train_p, val_p, meta_p = exporter.export_dataset(clean_records, metadata_payload)

        assert train_p.exists() and train_p.stat().st_size > 0, "train.jsonl must exist and be non-empty"
        assert val_p.exists() and val_p.stat().st_size > 0, "validation.jsonl must exist and be non-empty"
        assert meta_p.exists() and meta_p.stat().st_size > 0, "metadata.json must exist and be non-empty"

        with open(meta_p, "r", encoding="utf-8") as f:
            meta_json = json.load(f)

        assert meta_json.get("experiment") == "uncontrolled_recursive_baseline", f"Unexpected experiment: {meta_json.get('experiment')}"
        assert meta_json.get("source_generation") == 2, f"Unexpected source_generation: {meta_json.get('source_generation')}"
        assert meta_json.get("parent_student") == "generation_1", f"Unexpected parent_student: {meta_json.get('parent_student')}"
        assert meta_json.get("synthetic_ratio") == 1.0, f"Unexpected synthetic_ratio: {meta_json.get('synthetic_ratio')}"
        assert meta_json.get("anchor_ratio") == 0.0, f"Unexpected anchor_ratio: {meta_json.get('anchor_ratio')}"

        logger.info("[PASS] [9] & [10] Exported files verified. Metadata payload matches baseline control specification.")

        # --- Generate Report ---
        reporter = BaselineReportGenerator(config=config, logger=logger)
        summary_p = reporter.generate_report(val_report, metadata_payload)
        assert summary_p.exists()

        logger.info("================================================================================")
        logger.info(" ALL BASELINE VERIFICATION CHECKS [1-10] PASSED SUCCESSFULLY!")
        logger.info("================================================================================")
        return True

    except Exception as e:
        logger.error("Baseline Verification FAILED: %s", str(e), exc_info=True)
        return False


if __name__ == "__main__":
    success = verify_baseline_module()
    if not success:
        sys.exit(1)
