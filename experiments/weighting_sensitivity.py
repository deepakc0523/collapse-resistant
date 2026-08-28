"""
================================================================================
experiments/weighting_sensitivity.py
================================================================================

SCRS Weighting Sensitivity Analysis Script.

Evaluates Synthetic Collapse Risk Score (SCRS) variations across different
Representation vs Uncertainty weight combinations (e.g. 50:50, 60:40, 70:30).

Evaluates across:
  - Student-1
  - Student-2 Baseline
  - Student-2 Adaptive

Preserves 60:40 as global default configuration.

Usage:
  python -m experiments.weighting_sensitivity
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Setup root path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scrs.scrs_config import SCRSConfig
from scrs.scrs_engine import SCRSEngine
from scrs.weighting_engine import WeightingEngine

from experiments.utils import (
    save_json_report,
    save_text_summary,
    save_csv_report,
)
from experiments.visualization import plot_weighting_sensitivity

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("experiments.weighting_sensitivity")


def parse_args():
    parser = argparse.ArgumentParser(description="SCRS Weighting Sensitivity Analysis")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "research_results" / "final_validation" / "weighting_sensitivity",
        help="Output directory for weighting sensitivity artifacts",
    )
    parser.add_argument(
        "--weight-configs",
        nargs="+",
        default=["50:50", "60:40", "70:30"],
        help="Weighting configurations in format RepWeight:UncWeight (e.g. 50:50 60:40 70:30)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="weighting_sensitivity",
        help="Experiment identifier name",
    )
    return parser.parse_args()


def parse_weight_pair(pair_str: str) -> Tuple[float, float]:
    """Parses '60:40' string into tuple (0.60, 0.40)."""
    parts = pair_str.split(":")
    if len(parts) != 2:
        raise ValueError(f"Weight specification must be in format 'X:Y', got '{pair_str}'")
    rep_w = float(parts[0]) / 100.0 if float(parts[0]) > 1.0 else float(parts[0])
    unc_w = float(parts[1]) / 100.0 if float(parts[1]) > 1.0 else float(parts[1])
    
    total = rep_w + unc_w
    if abs(total - 1.0) > 1e-5:
        rep_w = rep_w / total
        unc_w = unc_w / total
    return rep_w, unc_w


def evaluate_model_sensitivity(
    model_name: str,
    base_rep_risk: float,
    base_unc_risk: float,
    weight_pairs: List[Tuple[float, float]]
) -> List[Dict[str, Any]]:
    """
    Computes SCRS and risk label for a given model across weighting pairs.
    """
    config = SCRSConfig()
    results = []

    for rep_w, unc_w in weight_pairs:
        config.representation_group_weight = rep_w
        config.uncertainty_group_weight = unc_w
        engine = SCRSEngine(config, logger=logger)
        
        scrs_val = engine.weighting_engine.compute_scrs(base_rep_risk, base_unc_risk)
        risk_label = engine.get_risk_label(scrs_val)

        results.append({
            "model_name": model_name,
            "representation_weight": rep_w,
            "uncertainty_weight": unc_w,
            "weight_label": f"{int(round(rep_w*100))}:{int(round(unc_w*100))}",
            "representation_risk": base_rep_risk,
            "uncertainty_risk": base_unc_risk,
            "scrs": float(scrs_val),
            "risk_label": risk_label
        })

    return results


def main() -> None:
    args = parse_args()
    logger.info("=" * 80)
    logger.info("SCRS WEIGHTING SENSITIVITY ANALYSIS")
    logger.info("Output directory: %s", args.output_dir)
    logger.info("=" * 80)

    weight_pairs = [parse_weight_pair(p) for p in args.weight_configs]
    
    # Include fine-grained grid for publication plot: 0.1 to 0.9 in 0.1 steps
    grid_pairs = list(weight_pairs)
    grid_steps = [(round(i * 0.1, 2), round((10 - i) * 0.1, 2)) for i in range(1, 10)]
    for gp in grid_steps:
        if not any(abs(gp[0] - wp[0]) < 1e-4 for wp in grid_pairs):
            grid_pairs.append(gp)
    grid_pairs = sorted(grid_pairs, key=lambda x: x[0])

    # Historical / calculated baseline risks for S1, S2-Baseline, S2-Adaptive
    model_risks = {
        "Student-1": (0.7250, 0.7660),
        "Student-2 Baseline": (0.8050, 0.8105),
        "Student-2 Adaptive": (0.7980, 0.8080),
    }

    all_results: Dict[str, List[Dict[str, Any]]] = {}
    plot_data: Dict[str, List[Tuple[float, float]]] = {}

    for model_name, (rep_r, unc_r) in model_risks.items():
        logger.info("Evaluating sensitivity for model '%s'...", model_name)
        m_results = evaluate_model_sensitivity(model_name, rep_r, unc_r, grid_pairs)
        all_results[model_name] = m_results
        plot_data[model_name] = [(r["representation_weight"], r["scrs"]) for r in m_results]

    # Save JSON summary
    summary_json = {
        "experiment_name": args.experiment_name,
        "default_weight": "60:40",
        "weight_configurations_evaluated": args.weight_configs,
        "sensitivity_results": all_results
    }
    json_path = args.output_dir / "weighting_sensitivity_report.json"
    save_json_report(summary_json, json_path)

    # Save CSV
    csv_headers = ["Model", "Weight_Config", "Rep_Weight", "Unc_Weight", "Rep_Risk", "Unc_Risk", "SCRS", "Risk_Label"]
    csv_rows = []
    for model_name, m_results in all_results.items():
        for r in m_results:
            csv_rows.append([
                model_name,
                r["weight_label"],
                f"{r['representation_weight']:.2f}",
                f"{r['uncertainty_weight']:.2f}",
                f"{r['representation_risk']:.4f}",
                f"{r['uncertainty_risk']:.4f}",
                f"{r['scrs']:.4f}",
                r["risk_label"]
            ])
    csv_path = args.output_dir / "weighting_sensitivity_summary.csv"
    save_csv_report(csv_headers, csv_rows, csv_path)

    # Save Text Summary
    txt_content = (
        "================================================================================\n"
        "                  SCRS WEIGHTING SENSITIVITY SUMMARY REPORT\n"
        "================================================================================\n\n"
        "Evaluated Weighting Ratios (Representation : Uncertainty):\n"
    )
    for model_name, m_results in all_results.items():
        txt_content += f"\n--- {model_name} ---\n"
        for r in m_results:
            if r["weight_label"] in args.weight_configs or r["weight_label"] == "60:40":
                txt_content += (
                    f"  Ratio {r['weight_label']} -> SCRS: {r['scrs']:.4f} [{r['risk_label']}] "
                    f"(Rep Contribution: {r['representation_weight']*r['representation_risk']:.4f}, "
                    f"Unc Contribution: {r['uncertainty_weight']*r['uncertainty_risk']:.4f})\n"
                )
    txt_content += "\n================================================================================\n"
    txt_path = args.output_dir / "weighting_sensitivity_summary.txt"
    save_text_summary(txt_content, txt_path)

    # Generate Publication Sensitivity Plot
    plot_path = args.output_dir / "scrs_weighting_sensitivity.png"
    plot_weighting_sensitivity(plot_data, plot_path)

    logger.info("=" * 80)
    logger.info("SCRS WEIGHTING SENSITIVITY ANALYSIS COMPLETED SUCCESSFULLY!")
    logger.info("  JSON Summary: %s", json_path)
    logger.info("  CSV Summary : %s", csv_path)
    logger.info("  TXT Summary : %s", txt_path)
    logger.info("  Plot output : %s", plot_path)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
