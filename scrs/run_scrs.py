"""
================================================================================
scrs/run_scrs.py
================================================================================

Main execution CLI entrypoint for the Synthetic Collapse Risk Score (SCRS) engine.

Usage:
    python -m scrs.run_scrs
"""

import sys
import logging
from pathlib import Path

from scrs.scrs_config import SCRSConfig
from scrs.scrs_engine import SCRSEngine
from scrs.scrs_report import SCRSReportGenerator
from scrs.visualization import SCRSVisualizer
from scrs.utils import get_scrs_logger, setup_utf8_terminal


def main() -> None:
    """Main execution entrypoint."""
    setup_utf8_terminal()
    logger = get_scrs_logger("scrs.run_scrs")

    logger.info("================================================================================")
    logger.info("              SYNTHETIC COLLAPSE RISK SCORE (SCRS) FUSION ENGINE")
    logger.info("================================================================================")

    config = SCRSConfig()
    engine = SCRSEngine(config, logger=logger)

    # 1. Execute Fusion Pipeline
    result = engine.compute()

    # 2. Generate Reports
    reporter = SCRSReportGenerator(config, logger=logger)
    json_path = reporter.generate_json_report(result)
    txt_path = reporter.generate_text_summary(result)

    # 3. Generate Visualizations
    visualizer = SCRSVisualizer(config, logger=logger)
    plot_paths = visualizer.generate_all_plots(result)

    # 4. Console Print Summary
    print("\n" + "=" * 80)
    print(f" SCRS SCORE         : {result.scrs:.4f}")
    print(f" RISK LEVEL LABEL   : {result.risk_label.upper()}")
    print(f" REPRESENTATION RISK: {result.representation_risk:.4f} (Weight: {result.group_weights['representation_group']*100:.0f}%)")
    print(f" UNCERTAINTY RISK   : {result.uncertainty_risk:.4f} (Weight: {result.group_weights['uncertainty_group']*100:.0f}%)")
    print("-" * 80)
    print(f" Report JSON        : {json_path}")
    print(f" Summary Text       : {txt_path}")
    print(" Plot Outputs:")
    for plot in plot_paths:
        print(f"   - {plot}")
    print("=" * 80 + "\n")

    logger.info("SCRS module execution completed successfully.")


if __name__ == "__main__":
    main()
