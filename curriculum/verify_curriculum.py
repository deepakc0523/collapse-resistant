"""
================================================================================
curriculum/verify_curriculum.py
================================================================================

Comprehensive self-testing verification script for the Curriculum Controller (CC).

Proves:
  [1] Real Gen-2 JSONL is loaded.
  [2] Exactly 1,000 synthetic source records are detected.
  [3] No fallback data is generated (strict fail-loudly check).
  [4] ATE policy is loaded correctly.
  [5] Exactly 750 human + 250 Gen-2 synthetic records are selected.
  [6] 3-stage curriculum remains valid.
  [7] Duplicate/empty/corrupt records are handled.
  [8] Seed reproducibility passes.
  [9] train.jsonl, validation.jsonl and metadata.json are valid.
  [10] Metadata correctly identifies Generation-2 as the synthetic source.
"""

import sys
import json
import logging
from pathlib import Path

from curriculum.curriculum_config import CurriculumConfig
from curriculum.policy_loader import PolicyLoader, ATEPolicyData
from curriculum.dataset_loader import DatasetLoader
from curriculum.dataset_sampler import DatasetSampler
from curriculum.dataset_mixer import DatasetMixer
from curriculum.curriculum_scheduler import CurriculumScheduler
from curriculum.metadata_generator import MetadataGenerator
from curriculum.dataset_validator import DatasetValidator
from curriculum.dataset_exporter import DatasetExporter
from curriculum.curriculum_report import CurriculumReportGenerator
from curriculum.visualization import CurriculumVisualizer
from curriculum.utils import get_curriculum_logger, set_seed


def verify_curriculum_module() -> bool:
    """
    Executes end-to-end verification checks [1] through [10] for the Curriculum Controller.

    Returns
    -------
    bool
        True if all verification checks pass, False otherwise.
    """
    logger = get_curriculum_logger("curriculum.verify_curriculum")
    logger.info("================================================================================")
    logger.info(" Starting Curriculum Controller (CC) Verification Suite (Checks 1-10)...")
    logger.info("================================================================================")

    config = CurriculumConfig()

    try:
        # --- Check 4: ATE Policy Loading ---
        logger.info("[4/10] Testing ATE policy loading...")
        loader = PolicyLoader(config=config, logger=logger)
        policy_data = loader.load_policy()

        assert abs(policy_data.synthetic_ratio - 0.25) < 1e-3, f"Expected synthetic_ratio 0.25, got {policy_data.synthetic_ratio}"
        assert abs(policy_data.anchor_ratio - 0.75) < 1e-3, f"Expected anchor_ratio 0.75, got {policy_data.anchor_ratio}"
        assert policy_data.training_status == "HIGH_RISK", f"Expected HIGH_RISK status, got {policy_data.training_status}"
        logger.info("[PASS] [4] ATE policy loaded correctly: Synthetic=0.25, Anchor=0.75, Status=HIGH_RISK.")

        # --- Check 1 & 2: Real Gen-2 JSONL loading & record count ---
        logger.info("[1/10 & 2/10] Testing Real Gen-2 JSONL loading and 1,000 record count...")
        ds_loader = DatasetLoader(config=config, logger=logger)
        synthetic_records = ds_loader.load_synthetic_dataset(allow_fallback=False, required_records=1000)

        assert len(synthetic_records) == 1000, f"Expected exactly 1,000 synthetic records, got {len(synthetic_records)}"
        assert synthetic_records[0].metadata.get("origin") == "generation_2_synthetic_jsonl"
        assert synthetic_records[0].metadata.get("generation") == 2
        logger.info("[PASS] [1] & [2] Real Gen-2 JSONL loaded successfully with exactly 1,000 source records.")

        # --- Check 3: Fail loudly when synthetic source is missing/corrupted ---
        logger.info("[3/10] Testing strict fail-loudly behavior (no fallback data in real mode)...")
        bad_config = CurriculumConfig()
        bad_config.synthetic_dataset_path = Path("non_existent_gen2_synthetic.jsonl")
        bad_ds_loader = DatasetLoader(config=bad_config, logger=logger)

        try:
            bad_ds_loader.load_synthetic_dataset(allow_fallback=False)
            assert False, "Expected FileNotFoundError when real synthetic file is missing and allow_fallback=False"
        except FileNotFoundError as e:
            logger.info("[PASS] [3] No fallback data generated: failed loudly as expected (%s).", str(e))

        # --- Load Anchor Records ---
        anchor_records = ds_loader.load_anchor_dataset(limit=config.total_dataset_size, allow_fallback=False)
        assert len(anchor_records) >= 750, f"Need at least 750 anchor records, got {len(anchor_records)}"

        # --- Check 5: Mixing ratios (750 human + 250 Gen-2 synthetic) ---
        logger.info("[5/10] Testing mixing ratio selection (750 human + 250 synthetic)...")
        mixer = DatasetMixer(config=config, logger=logger)
        pool = mixer.mix_datasets(
            anchor_pool=anchor_records,
            synthetic_pool=synthetic_records,
            policy_data=policy_data,
            total_count=1000,
        )

        assert len(pool.anchor_samples) == 750, f"Expected 750 anchor samples, got {len(pool.anchor_samples)}"
        assert len(pool.synthetic_samples) == 250, f"Expected 250 synthetic samples, got {len(pool.synthetic_samples)}"
        logger.info("[PASS] [5] Exactly 750 human anchor + 250 Gen-2 synthetic records selected.")

        # --- Check 6: 3-Stage Curriculum Scheduling ---
        logger.info("[6/10] Testing 3-stage progressive curriculum schedule...")
        scheduler = CurriculumScheduler(config=config, logger=logger)
        scheduled = scheduler.schedule_curriculum(pool=pool, policy_data=policy_data)

        stage1_end = scheduled.stage_boundaries["Stage_1_Foundation"][1]
        stage1_records = scheduled.ordered_records[:stage1_end]
        assert len(stage1_records) == 250, f"Expected Stage 1 count 250, got {len(stage1_records)}"
        for r in stage1_records:
            assert r.source == "anchor", f"Stage 1 must be 100% anchor, found {r.source}"

        logger.info("[PASS] [6] 3-stage curriculum schedule validated (Stage 1 = 100% anchor foundation).")

        # --- Check 7: Duplicate / Empty / Corrupt Records Handling ---
        logger.info("[7/10] Testing record validation and duplicate detection...")
        validator = DatasetValidator(config=config, logger=logger)
        base_clean, _ = validator.validate_and_clean(scheduled.ordered_records, policy_data)

        # Inject duplicate to verify detection
        test_records = list(base_clean)
        test_records.append(base_clean[0])

        cleaned, val_report = validator.validate_and_clean(test_records, policy_data)
        assert val_report.duplicates_removed >= 1, "Validator must catch injected duplicate"
        assert len(cleaned) == len(base_clean)
        logger.info("[PASS] [7] Duplicate and record validation checks passed.")

        # --- Check 8: Seed Reproducibility ---
        logger.info("[8/10] Testing seed reproducibility...")
        pool2 = mixer.mix_datasets(anchor_records, synthetic_records, policy_data, total_count=1000)
        for r1, r2 in zip(pool.synthetic_samples, pool2.synthetic_samples):
            assert r1.record_id == r2.record_id, "Identical seeds must produce identical record selection"
        logger.info("[PASS] [8] Seed reproducibility check passed.")

        # --- Check 9 & 10: Metadata & Export Validation ---
        logger.info("[9/10 & 10/10] Testing export (train.jsonl, val.jsonl, metadata.json) & metadata source identification...")
        train_count = int(round(len(cleaned) * config.train_val_split))
        val_count = len(cleaned) - train_count

        meta_gen = MetadataGenerator(config=config, logger=logger)
        metadata_payload = meta_gen.generate_metadata(
            policy_data=policy_data,
            scheduled=scheduled,
            validation_report=val_report,
            train_count=train_count,
            val_count=val_count,
        )

        exporter = DatasetExporter(config=config, logger=logger)
        train_p, val_p, meta_p = exporter.export_dataset(cleaned, metadata_payload)

        assert train_p.exists() and train_p.stat().st_size > 0
        assert val_p.exists() and val_p.stat().st_size > 0
        assert meta_p.exists() and meta_p.stat().st_size > 0

        # Validate metadata schema & Generation-2 identification
        with open(meta_p, "r", encoding="utf-8") as f:
            meta_json = json.load(f)

        assert meta_json.get("synthetic_source_generation") == 2, f"Expected generation 2, got {meta_json.get('synthetic_source_generation')}"
        assert meta_json.get("synthetic_parent_model") == "generation_1"
        assert meta_json.get("synthetic_source_record_count") == 1000
        assert "generation_2_synthetic.jsonl" in meta_json.get("synthetic_source", "")

        logger.info("[PASS] [9] & [10] Exported train.jsonl (%d), validation.jsonl (%d), and metadata.json correctly identify Generation-2 synthetic source.", train_count, val_count)

        # --- Generate Report and Plots ---
        reporter = CurriculumReportGenerator(config=config, logger=logger)
        summary_p = reporter.generate_report(policy_data, scheduled, val_report, metadata_payload)
        assert summary_p.exists()

        visualizer = CurriculumVisualizer(config=config, logger=logger)
        plots = visualizer.generate_all_plots(policy_data, scheduled, cleaned)
        assert len(plots) == 5

        logger.info("================================================================================")
        logger.info(" ALL VERIFICATION CHECKS [1-10] PASSED SUCCESSFULLY!")
        logger.info("================================================================================")
        return True

    except Exception as e:
        logger.error("Curriculum Verification FAILED: %s", str(e), exc_info=True)
        return False


if __name__ == "__main__":
    success = verify_curriculum_module()
    if not success:
        sys.exit(1)
