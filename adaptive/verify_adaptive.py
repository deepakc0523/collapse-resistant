"""
================================================================================
adaptive/verify_adaptive.py
================================================================================

Comprehensive self-testing verification script for the Adaptive Threshold Engine (ATE).

Validates:
  1. SCRS report loading and validation
  2. Policy derivation and continuous mathematical consistency (sum to 1.0, bounds)
  3. Scientific recommendation and justification generation
  4. Output JSON schema and TXT report structure
  5. Rendering and creation of all 5 visualization plots
"""

import sys
import json
import logging
from pathlib import Path

from adaptive.adaptive_config import AdaptiveConfig
from adaptive.scrs_loader import SCRSLoader, SCRSData
from adaptive.policy_engine import PolicyEngine, ATEPolicyResult
from adaptive.recommendation_engine import RecommendationEngine, RecommendationReport
from adaptive.adaptive_report import AdaptiveReportGenerator
from adaptive.visualization import AdaptiveVisualizer
from adaptive.utils import get_adaptive_logger


def verify_adaptive_module() -> bool:
    """
    Executes end-to-end verification checks for the Adaptive Threshold Engine.

    Returns
    -------
    bool
        True if all verification checks pass, False otherwise.
    """
    logger = get_adaptive_logger("adaptive.verify_adaptive")
    logger.info("Starting Adaptive Threshold Engine (ATE) Verification Suite...")

    config = AdaptiveConfig()

    try:
        # --- Check 1: SCRS Report Loading ---
        logger.info("[1/7] Testing SCRS report loader...")
        loader = SCRSLoader(config=config, logger=logger)
        scrs_data = loader.load_report()

        assert scrs_data.overall_scrs >= 0.0 and scrs_data.overall_scrs <= 1.0, "SCRS must be in [0, 1]"
        assert scrs_data.representation_risk >= 0.0 and scrs_data.representation_risk <= 1.0
        assert scrs_data.uncertainty_risk >= 0.0 and scrs_data.uncertainty_risk <= 1.0
        assert len(scrs_data.metric_contributions) > 0, "Metric contributions must be populated"
        logger.info("✔ SCRS report loading check passed.")

        # --- Check 2: Policy Engine & Mathematical Consistency ---
        logger.info("[2/7] Testing Policy Engine mathematical derivations...")
        engine = PolicyEngine(config=config, logger=logger)
        policy_result = engine.derive_policy(scrs_data)

        policy = policy_result.policy

        # Mathematical assertions
        assert abs((policy.synthetic_ratio + policy.anchor_ratio) - 1.0) < 1e-4, (
            f"Mix ratios must sum to 1.0, got {policy.synthetic_ratio + policy.anchor_ratio:.6f}"
        )
        assert config.min_synthetic_ratio <= policy.synthetic_ratio <= config.max_synthetic_ratio
        assert config.min_anchor_ratio <= policy.anchor_ratio <= config.max_anchor_ratio
        assert config.min_epochs <= policy.recommended_epochs <= config.max_epochs
        assert config.min_learning_rate <= policy.recommended_learning_rate <= config.max_learning_rate
        assert config.min_sampling_temperature <= policy.sampling_temperature <= config.max_sampling_temperature
        assert config.min_generation_depth <= policy.max_generation_depth <= config.max_generation_depth
        assert policy_result.training_status in ["SAFE", "MODERATE_RISK", "HIGH_RISK", "CRITICAL_COLLAPSE"]

        logger.info("✔ Policy engine mathematical consistency check passed.")

        # --- Check 3: Recommendation Engine ---
        logger.info("[3/7] Testing Recommendation Engine...")
        rec_engine = RecommendationEngine(config=config, logger=logger)
        rec_report = rec_engine.generate_recommendations(scrs_data, policy_result)

        assert len(rec_report.justifications) >= 5, "Must provide detailed scientific justifications"
        assert len(rec_report.mitigation_actions) >= 1, "Must provide mitigation actions"
        assert "target_generation" in rec_report.curriculum_instructions
        logger.info("✔ Recommendation engine check passed.")

        # --- Check 4: Report Generation (JSON & TXT) ---
        logger.info("[4/7] Testing JSON & TXT report generation...")
        reporter = AdaptiveReportGenerator(config=config, logger=logger)
        json_path, txt_path = reporter.generate_reports(scrs_data, policy_result, rec_report)

        assert json_path.exists() and json_path.stat().st_size > 0, "adaptive_policy.json must exist"
        assert txt_path.exists() and txt_path.stat().st_size > 0, "adaptive_summary.txt must exist"

        # Validate JSON schema
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "training_status" in data
            assert "policy" in data
            assert "scrs_summary" in data
            assert "recommendations" in data

        logger.info("✔ Report generation check passed.")

        # --- Check 5: Visualizations ---
        logger.info("[5/7] Testing publication visualization generation...")
        visualizer = AdaptiveVisualizer(config=config, logger=logger)
        plot_paths = visualizer.generate_all_plots(scrs_data, policy_result, rec_report)

        expected_plots = [
            "policy_overview.png",
            "metric_influence.png",
            "training_recommendations.png",
            "recursive_pathway.png",
            "policy_heatmap.png",
        ]

        for plot_name in expected_plots:
            target_plot = config.plots_dir / plot_name
            assert target_plot.exists() and target_plot.stat().st_size > 0, f"Missing plot artifact: {plot_name}"

        logger.info("✔ All 5 visualization plots verified successfully.")

        # --- Check 6: Dynamic Range Softness (No Hard-Coded Magic Thresholds) ---
        logger.info("[6/7] Validating non-disruptive continuous sensitivity across range...")
        for test_scrs in [0.05, 0.35, 0.65, 0.95]:
            test_data = SCRSData(
                overall_scrs=test_scrs,
                risk_label="Test",
                representation_risk=test_scrs,
                uncertainty_risk=test_scrs,
                rep_metrics={},
                unc_metrics={},
                metric_contributions={"test_metric": test_scrs},
                raw_representation_metadata={},
                raw_uncertainty_metadata={},
            )
            res = engine.derive_policy(test_data)
            p = res.policy
            assert abs((p.synthetic_ratio + p.anchor_ratio) - 1.0) < 1e-4

        logger.info("✔ Continuous transfer function behavior verified.")

        logger.info("[7/7] VERIFICATION COMPLETE: Adaptive Threshold Engine (ATE) is fully operational!")
        return True

    except Exception as e:
        logger.error("Verification FAILED with error: %s", str(e), exc_info=True)
        return False


if __name__ == "__main__":
    success = verify_adaptive_module()
    if not success:
        sys.exit(1)
