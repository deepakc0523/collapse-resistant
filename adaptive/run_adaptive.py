"""
================================================================================
adaptive/run_adaptive.py
================================================================================

Main pipeline execution script for the Adaptive Threshold Engine (ATE).

Pipeline:
  scrs_out/scrs_report.json -> SCRSLoader -> PolicyEngine -> RecommendationEngine
    -> AdaptiveReportGenerator -> AdaptiveVisualizer -> adaptive_out/ & plots/
"""

import sys
import logging
from pathlib import Path

from adaptive.adaptive_config import AdaptiveConfig
from adaptive.scrs_loader import SCRSLoader
from adaptive.policy_engine import PolicyEngine
from adaptive.recommendation_engine import RecommendationEngine
from adaptive.adaptive_report import AdaptiveReportGenerator
from adaptive.visualization import AdaptiveVisualizer
from adaptive.utils import get_adaptive_logger


def run_adaptive_pipeline() -> None:
    """Runs the end-to-end Adaptive Threshold Engine pipeline."""
    logger = get_adaptive_logger("adaptive.run_adaptive")
    logger.info("================================================================================")
    logger.info(" Starting Adaptive Threshold Engine (ATE) Execution...")
    logger.info("================================================================================")

    config = AdaptiveConfig()

    # 1. Load SCRS Report
    loader = SCRSLoader(config=config, logger=logger)
    scrs_data = loader.load_report()

    # 2. Derive Continuous Hyperparameter Policy
    engine = PolicyEngine(config=config, logger=logger)
    policy_result = engine.derive_policy(scrs_data)

    # 3. Synthesize Scientific Recommendations & Curriculum Instructions
    rec_engine = RecommendationEngine(config=config, logger=logger)
    rec_report = rec_engine.generate_recommendations(scrs_data, policy_result)

    # 4. Generate JSON and TXT Reports
    reporter = AdaptiveReportGenerator(config=config, logger=logger)
    json_path, txt_path = reporter.generate_reports(scrs_data, policy_result, rec_report)

    # 5. Generate Publication Visualizations
    visualizer = AdaptiveVisualizer(config=config, logger=logger)
    plots = visualizer.generate_all_plots(scrs_data, policy_result, rec_report)

    logger.info("--------------------------------------------------------------------------------")
    logger.info(" ATE Pipeline Execution Summary:")
    logger.info("  • Training Status     : %s", policy_result.training_status)
    logger.info("  • Synthetic Ratio     : %.4f", policy_result.policy.synthetic_ratio)
    logger.info("  • Anchor Ratio        : %.4f", policy_result.policy.anchor_ratio)
    logger.info("  • Recommended Epochs  : %d", policy_result.policy.recommended_epochs)
    logger.info("  • Learning Rate       : %.2e", policy_result.policy.recommended_learning_rate)
    logger.info("  • Sampling Temp       : %.4f", policy_result.policy.sampling_temperature)
    logger.info("  • Policy JSON Artifact: %s", json_path)
    logger.info("  • Text Summary        : %s", txt_path)
    logger.info("  • Visualizations Generated: %d plots in %s", len(plots), config.plots_dir)
    logger.info("================================================================================")


if __name__ == "__main__":
    run_adaptive_pipeline()
