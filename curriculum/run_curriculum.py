"""
================================================================================
curriculum/run_curriculum.py
================================================================================

Main pipeline execution script for the Curriculum Controller (CC).

Pipeline:
  adaptive_policy.json + Anchor Data + Synthetic Data -> PolicyLoader -> DatasetLoader
    -> DatasetMixer -> CurriculumScheduler -> DatasetValidator -> MetadataGenerator
    -> DatasetExporter -> CurriculumVisualizer -> curriculum_out/generation_2/ & plots/
"""

import sys
import logging
from pathlib import Path

from curriculum.curriculum_config import CurriculumConfig
from curriculum.policy_loader import PolicyLoader
from curriculum.dataset_loader import DatasetLoader
from curriculum.dataset_mixer import DatasetMixer
from curriculum.curriculum_scheduler import CurriculumScheduler
from curriculum.metadata_generator import MetadataGenerator
from curriculum.dataset_validator import DatasetValidator
from curriculum.dataset_exporter import DatasetExporter
from curriculum.curriculum_report import CurriculumReportGenerator
from curriculum.visualization import CurriculumVisualizer
from curriculum.utils import get_curriculum_logger


def run_curriculum_pipeline() -> None:
    """Runs the end-to-end Curriculum Controller pipeline."""
    logger = get_curriculum_logger("curriculum.run_curriculum")
    logger.info("================================================================================")
    logger.info(" Starting Curriculum Controller (CC) Execution...")
    logger.info("================================================================================")

    config = CurriculumConfig()

    # 1. Load ATE Policy
    policy_loader = PolicyLoader(config=config, logger=logger)
    policy_data = policy_loader.load_policy()

    # 2. Load Raw Datasets (Strict Mode: allow_fallback=False)
    dataset_loader = DatasetLoader(config=config, logger=logger)
    anchor_records = dataset_loader.load_anchor_dataset(limit=config.total_dataset_size, allow_fallback=False)
    synthetic_records = dataset_loader.load_synthetic_dataset(
        limit=config.total_dataset_size, allow_fallback=False, required_records=1000
    )

    # 3. Mix Datasets Proportionally
    mixer = DatasetMixer(config=config, logger=logger)
    pool = mixer.mix_datasets(
        anchor_pool=anchor_records,
        synthetic_pool=synthetic_records,
        policy_data=policy_data,
        total_count=config.total_dataset_size,
    )

    # 4. Schedule 3-Stage Progressive Curriculum
    scheduler = CurriculumScheduler(config=config, logger=logger)
    scheduled = scheduler.schedule_curriculum(pool=pool, policy_data=policy_data)

    # 5. Validate & Clean Records
    validator = DatasetValidator(config=config, logger=logger)
    cleaned_records, val_report = validator.validate_and_clean(
        records=scheduled.ordered_records, policy_data=policy_data
    )

    # 6. Generate Reproducibility Metadata
    train_count = int(round(len(cleaned_records) * config.train_val_split))
    val_count = len(cleaned_records) - train_count

    meta_gen = MetadataGenerator(config=config, logger=logger)
    metadata_payload = meta_gen.generate_metadata(
        policy_data=policy_data,
        scheduled=scheduled,
        validation_report=val_report,
        train_count=train_count,
        val_count=val_count,
    )

    # 7. Export Dataset Artifacts (train.jsonl, val.jsonl, metadata.json)
    exporter = DatasetExporter(config=config, logger=logger)
    train_p, val_p, meta_p = exporter.export_dataset(
        records=cleaned_records, metadata_payload=metadata_payload
    )

    # 8. Generate Summary Text Report
    reporter = CurriculumReportGenerator(config=config, logger=logger)
    summary_p = reporter.generate_report(
        policy_data=policy_data,
        scheduled=scheduled,
        validation_report=val_report,
        metadata=metadata_payload,
    )

    # 9. Render Publication Visualizations
    visualizer = CurriculumVisualizer(config=config, logger=logger)
    plots = visualizer.generate_all_plots(
        policy_data=policy_data, scheduled=scheduled, records=cleaned_records
    )

    logger.info("--------------------------------------------------------------------------------")
    logger.info(" Curriculum Controller Pipeline Execution Summary:")
    logger.info("  • Target Generation   : Generation-2")
    logger.info("  • Training Status     : %s", policy_data.training_status)
    logger.info("  • Total Exported Size : %d samples", len(cleaned_records))
    logger.info("  • Synthetic Ratio     : %.4f", policy_data.synthetic_ratio)
    logger.info("  • Anchor Ratio        : %.4f", policy_data.anchor_ratio)
    logger.info("  • Export Directory    : %s", config.generation_output_dir)
    logger.info("  • Train Dataset       : %s", train_p)
    logger.info("  • Validation Dataset  : %s", val_p)
    logger.info("  • Metadata File       : %s", meta_p)
    logger.info("  • Summary Text Report : %s", summary_p)
    logger.info("  • Plots Generated     : %d visual artifacts in %s", len(plots), config.plots_dir)
    logger.info("================================================================================")


if __name__ == "__main__":
    run_curriculum_pipeline()
