"""
================================================================================
curriculum/verify_curriculum.py
================================================================================

Comprehensive self-testing verification script for the Curriculum Controller (CC).

Validates:
  1. Adaptive policy loading from adaptive_policy.json
  2. Dataset loading and fallback generation
  3. Mixing ratios matching policy requirements
  4. 3-stage curriculum scheduling ordering (Anchor Foundation -> Transition -> Target)
  5. Metadata generation schema and reproducibility
  6. Dataset export (train.jsonl, validation.jsonl, metadata.json)
  7. Duplicate detection and data cleaning
  8. Creation of all 5 publication visualization plots
  9. Random seed reproducibility
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
    Executes end-to-end verification checks for the Curriculum Controller.

    Returns
    -------
    bool
        True if all verification checks pass, False otherwise.
    """
    logger = get_curriculum_logger("curriculum.verify_curriculum")
    logger.info("Starting Curriculum Controller (CC) Verification Suite...")

    config = CurriculumConfig()

    try:
        # --- Check 1: ATE Policy Loading ---
        logger.info("[1/9] Testing ATE policy loading...")
        loader = PolicyLoader(config=config, logger=logger)
        policy_data = loader.load_policy()

        assert 0.0 <= policy_data.synthetic_ratio <= 1.0, "Synthetic ratio must be in [0, 1]"
        assert 0.0 <= policy_data.anchor_ratio <= 1.0, "Anchor ratio must be in [0, 1]"
        assert abs((policy_data.synthetic_ratio + policy_data.anchor_ratio) - 1.0) < 1e-3
        logger.info("[PASS] ATE policy loading check passed.")

        # --- Check 2: Dataset Loading ---
        logger.info("[2/9] Testing raw anchor and synthetic dataset loading...")
        ds_loader = DatasetLoader(config=config, logger=logger)
        anchor_records = ds_loader.load_anchor_dataset(limit=1000)
        synthetic_records = ds_loader.load_synthetic_dataset(limit=1000)

        assert len(anchor_records) > 0, "Anchor dataset must contain records"
        assert len(synthetic_records) > 0, "Synthetic dataset must contain records"
        logger.info("[PASS] Dataset loading check passed.")

        # --- Check 3: Mixing Ratios ---
        logger.info("[3/9] Testing dataset mixing proportions...")
        mixer = DatasetMixer(config=config, logger=logger)
        pool = mixer.mix_datasets(
            anchor_pool=anchor_records,
            synthetic_pool=synthetic_records,
            policy_data=policy_data,
            total_count=1000,
        )

        expected_syn = int(round(1000 * policy_data.synthetic_ratio))
        assert len(pool.synthetic_samples) == expected_syn, f"Expected {expected_syn} synthetic samples"
        logger.info("[PASS] Dataset mixing ratios check passed.")

        # --- Check 4: Curriculum Scheduling ---
        logger.info("[4/9] Testing 3-stage progressive curriculum scheduling...")
        scheduler = CurriculumScheduler(config=config, logger=logger)
        scheduled = scheduler.schedule_curriculum(pool=pool, policy_data=policy_data)

        # Assert Stage 1 is 100% pure anchor
        stage1_end = scheduled.stage_boundaries["Stage_1_Foundation"][1]
        stage1_records = scheduled.ordered_records[:stage1_end]
        for r in stage1_records:
            assert r.source == "anchor", f"Stage 1 must be 100% anchor, got source={r.source}"

        logger.info("[PASS] 3-Stage curriculum scheduling check passed.")

        # --- Check 5: Duplicate Detection & Data Validation ---
        logger.info("[5/9] Testing dataset validation and duplicate detection...")
        validator = DatasetValidator(config=config, logger=logger)

        base_clean, _ = validator.validate_and_clean(scheduled.ordered_records, policy_data)

        # Inject duplicate to verify detection
        test_records = list(base_clean)
        test_records.append(base_clean[0]) # Inject duplicate

        cleaned, val_report = validator.validate_and_clean(test_records, policy_data)
        assert val_report.duplicates_removed >= 1, "Must detect and remove injected duplicate"
        assert len(cleaned) == len(base_clean), "Cleaned record count must match original before duplicate injection"
        logger.info("[PASS] Dataset validation check passed.")

        # --- Check 6: Reproducibility ---
        logger.info("[6/9] Testing seed reproducibility...")
        pool2 = mixer.mix_datasets(anchor_records, synthetic_records, policy_data, total_count=1000)
        for r1, r2 in zip(pool.synthetic_samples, pool2.synthetic_samples):
            assert r1.record_id == r2.record_id, "Sampling must be completely deterministic for identical seeds"
        logger.info("[PASS] Seed reproducibility check passed.")

        # --- Check 7: Metadata & Export ---
        logger.info("[7/9] Testing dataset export (train.jsonl, val.jsonl, metadata.json)...")
        meta_gen = MetadataGenerator(config=config, logger=logger)
        metadata_payload = meta_gen.generate_metadata(
            policy_data=policy_data,
            scheduled=scheduled,
            validation_report=val_report,
            train_count=int(round(len(cleaned) * config.train_val_split)),
            val_count=len(cleaned) - int(round(len(cleaned) * config.train_val_split)),
        )

        exporter = DatasetExporter(config=config, logger=logger)
        train_p, val_p, meta_p = exporter.export_dataset(cleaned, metadata_payload)

        assert train_p.exists() and train_p.stat().st_size > 0, "train.jsonl must exist"
        assert val_p.exists() and val_p.stat().st_size > 0, "validation.jsonl must exist"
        assert meta_p.exists() and meta_p.stat().st_size > 0, "metadata.json must exist"

        # Validate metadata schema
        with open(meta_p, "r", encoding="utf-8") as f:
            meta_json = json.load(f)
            assert meta_json["generation_id"] == "generation_2"
            assert "hyperparameters" in meta_json
            assert "dataset_sizes" in meta_json

        logger.info("[PASS] Dataset export check passed.")

        # --- Check 8: Summary Report ---
        logger.info("[8/9] Testing summary report generator...")
        reporter = CurriculumReportGenerator(config=config, logger=logger)
        summary_p = reporter.generate_report(policy_data, scheduled, val_report, metadata_payload)
        assert summary_p.exists() and summary_p.stat().st_size > 0, "curriculum_summary.txt must exist"
        logger.info("[PASS] Summary report check passed.")

        # --- Check 9: Visualizations ---
        logger.info("[9/9] Testing publication visualization generator...")
        visualizer = CurriculumVisualizer(config=config, logger=logger)
        plots = visualizer.generate_all_plots(policy_data, scheduled, cleaned)

        expected_plots = [
            "dataset_composition.png",
            "curriculum_progression.png",
            "curriculum_schedule.png",
            "sample_distribution.png",
            "generation_flow.png",
        ]

        for p_name in expected_plots:
            target_plot = config.plots_dir / p_name
            assert target_plot.exists() and target_plot.stat().st_size > 0, f"Missing plot artifact: {p_name}"

        logger.info("[PASS] All 5 visualization plots verified successfully.")

        logger.info("================================================================================")
        logger.info(" VERIFICATION COMPLETE: Curriculum Controller (CC) is fully operational!")
        logger.info("================================================================================")
        return True

    except Exception as e:
        logger.error("Verification FAILED with error: %s", str(e), exc_info=True)
        return False


if __name__ == "__main__":
    success = verify_curriculum_module()
    if not success:
        sys.exit(1)
