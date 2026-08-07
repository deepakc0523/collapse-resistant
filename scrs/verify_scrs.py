"""
================================================================================
scrs/verify_scrs.py
================================================================================

Verification script for the Synthetic Collapse Risk Score (SCRS) module.

Validates:
  ✓ Probe report loads
  ✓ Ensemble report loads
  ✓ All metrics normalize correctly to [0, 1]
  ✓ Weighting engine functions correctly
  ✓ SCRS score remains strictly within [0, 1]
  ✓ Reports are generated properly
  ✓ All 5 visualizations are generated
"""

import sys
import logging
from pathlib import Path

from scrs.scrs_config import SCRSConfig
from scrs.scrs_engine import SCRSEngine
from scrs.scrs_report import SCRSReportGenerator
from scrs.visualization import SCRSVisualizer
from scrs.utils import get_scrs_logger, setup_utf8_terminal


def run_verification() -> bool:
    """Executes verification checks on SCRS module components."""
    setup_utf8_terminal()
    logger = get_scrs_logger("scrs.verify")
    config = SCRSConfig()

    print("=" * 80)
    print("                    SCRS MODULE VERIFICATION SUITE")
    print("=" * 80)

    all_passed = True

    # Check 1: Probe Report Loading
    try:
        engine = SCRSEngine(config, logger=logger)
        result = engine.compute()
        print(" [PASS] ✓ Upstream Probe & Ensemble reports loaded successfully.")
    except Exception as e:
        print(f" [FAIL] ✗ Failed loading upstream reports: {e}")
        all_passed = False
        return False

    # Check 2: Metric Normalization Range [0, 1]
    norm_pass = True
    for group_name, metric_dict in [
        ("Representation", result.representation_metrics),
        ("Uncertainty", result.uncertainty_metrics),
    ]:
        for k, v in metric_dict.items():
            if not (0.0 <= v <= 1.0):
                print(f" [FAIL] ✗ Metric '{group_name}.{k}' out of range: {v}")
                norm_pass = False
    if norm_pass:
        print(" [PASS] ✓ All raw metrics normalized strictly to Risk Scale [0.0, 1.0].")
    else:
        all_passed = False

    # Check 3: Weighting Engine Sums & Math
    weight_pass = True
    rep_w_sum = sum(result.representation_weights.values())
    unc_w_sum = sum(result.uncertainty_weights.values())
    group_w_sum = sum(result.group_weights.values())

    if abs(rep_w_sum - 1.0) > 1e-4 or abs(unc_w_sum - 1.0) > 1e-4 or abs(group_w_sum - 1.0) > 1e-4:
        print(f" [FAIL] ✗ Weight summation error: rep={rep_w_sum}, unc={unc_w_sum}, group={group_w_sum}")
        weight_pass = False
    
    if weight_pass:
        print(" [PASS] ✓ Weighting engine validated (all weights sum to 1.0).")
    else:
        all_passed = False

    # Check 4: SCRS Bounded Range [0, 1]
    if 0.0 <= result.scrs <= 1.0:
        print(f" [PASS] ✓ Overall SCRS is bounded in [0, 1]: Score = {result.scrs:.4f} ({result.risk_label}).")
    else:
        print(f" [FAIL] ✗ SCRS score out of bounds: {result.scrs}")
        all_passed = False

    # Check 5: Report Generation
    try:
        report_gen = SCRSReportGenerator(config, logger=logger)
        json_path = report_gen.generate_json_report(result)
        txt_path = report_gen.generate_text_summary(result)

        if json_path.is_file() and txt_path.is_file():
            print(" [PASS] ✓ Reports generated (scrs_report.json, scrs_summary.txt).")
        else:
            print(" [FAIL] ✗ Report files missing after generation.")
            all_passed = False
    except Exception as e:
        print(f" [FAIL] ✗ Error generating reports: {e}")
        all_passed = False

    # Check 6: Visualization Generation
    try:
        visualizer = SCRSVisualizer(config, logger=logger)
        plot_paths = visualizer.generate_all_plots(result)
        
        missing_plots = [p for p in plot_paths if not p.is_file()]
        if not missing_plots and len(plot_paths) == 5:
            print(" [PASS] ✓ All 5 publication-quality visualizations generated in scrs_out/plots/.")
        else:
            print(f" [FAIL] ✗ Missing plot files: {missing_plots}")
            all_passed = False
    except Exception as e:
        print(f" [FAIL] ✗ Error generating visualizations: {e}")
        all_passed = False

    print("=" * 80)
    if all_passed:
        print(" [SUMMARY] SUCCESS: All SCRS verification checks passed!")
    else:
        print(" [SUMMARY] FAILURE: Some verification checks failed.")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
