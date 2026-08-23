"""
================================================================================
baseline/run_baseline.py
================================================================================

Main pipeline execution script for the Student-2 Baseline Dataset Builder.

Pipeline:
  data/synthetic/generation_2/generation_2_synthetic.jsonl
    -> BaselineLoader
    -> BaselineValidator
    -> MetadataGenerator
    -> BaselineExporter
    -> BaselineReportGenerator
    -> baseline_out/generation_2/
"""

import sys
import logging
from pathlib import Path

from baseline.baseline_config import BaselineConfig
from baseline.baseline_loader import BaselineLoader
from baseline.baseline_validator import BaselineValidator
from baseline.metadata_generator import MetadataGenerator
from baseline.baseline_exporter import BaselineExporter
from baseline.baseline_report import BaselineReportGenerator
from baseline.utils import get_baseline_logger


def run_baseline_pipeline() -> None:
    """Runs the end-to-end Student-2 Baseline Dataset Builder pipeline."""
    logger = get_baseline_logger("baseline.run_baseline")
    logger.info("================================================================================")
    logger.info(" Starting Student-2 Control Baseline Dataset Builder...")
    logger.info("================================================================================")

    config = BaselineConfig()

    # 1. Load Real Generation-2 Synthetic Records (Strict Mode: allow_fallback=False)
    loader = BaselineLoader(config=config, logger=logger)
    raw_records = loader.load_synthetic_dataset(allow_fallback=False, required_records=1000)

    # 2. Validate & Clean Records
    validator = BaselineValidator(config=config, logger=logger)
    clean_records, val_report = validator.validate_and_clean(records=raw_records)

    # 3. Calculate Train/Val Split Counts
    train_count = int(round(len(clean_records) * config.train_val_split))
    val_count = len(clean_records) - train_count

    # 4. Generate Metadata
    meta_gen = MetadataGenerator(config=config, logger=logger)
    metadata_payload = meta_gen.generate_metadata(
        validation_report=val_report,
        train_count=train_count,
        val_count=val_count,
    )

    # 5. Export Dataset Artifacts (train.jsonl, validation.jsonl, metadata.json)
    exporter = BaselineExporter(config=config, logger=logger)
    train_p, val_p, meta_p = exporter.export_dataset(
        records=clean_records, metadata_payload=metadata_payload
    )

    # 6. Generate Summary Text Report
    reporter = BaselineReportGenerator(config=config, logger=logger)
    summary_p = reporter.generate_report(
        validation_report=val_report,
        metadata=metadata_payload,
    )

    logger.info("--------------------------------------------------------------------------------")
    logger.info(" Student-2 Control Baseline Execution Summary:")
    logger.info("  • Experiment ID       : uncontrolled_recursive_baseline")
    logger.info("  • Source Generation   : Generation-2")
    logger.info("  • Total Exported Size : %d samples", len(clean_records))
    logger.info("  • Synthetic Ratio     : 1.0000 (100% Synthetic)")
    logger.info("  • Anchor Ratio        : 0.0000 (0% Human Anchor)")
    logger.info("  • Export Directory    : %s", config.generation_output_dir)
    logger.info("  • Train Dataset       : %s", train_p)
    logger.info("  • Validation Dataset  : %s", val_p)
    logger.info("  • Metadata File       : %s", meta_p)
    logger.info("  • Summary Text Report : %s", summary_p)
    logger.info("================================================================================")


if __name__ == "__main__":
    run_baseline_pipeline()
