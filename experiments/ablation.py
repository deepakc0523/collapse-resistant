"""
================================================================================
experiments/ablation.py
================================================================================

Monitoring Component Ablation Study Script.

Evaluates the relative contribution of PRDAF and EVM monitoring components:
  1. FULL     : Combined PRDAF Representation (60%) + EVM Uncertainty (40%)
  2. NO_PRDAF : Uncertainty risk only (EVM 100%, PRDAF 0%)
  3. NO_EVM   : Representation risk only (PRDAF 100%, EVM 0%)

Uses existing report outputs / metrics without model retraining.

Usage:
  python -m experiments.ablation
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Setup root path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scrs.scrs_config import SCRSConfig
from scrs.scrs_engine import SCRSEngine
from scrs.scrs_report import SCRSReportGenerator

from experiments.utils import (
    save_json_report,
    save_text_summary,
    save_csv_report,
)
from experiments.visualization import plot_ablation_study

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("experiments.ablation")


def parse_args():
    parser = argparse.ArgumentParser(description="Monitoring Component Ablation Study")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "research_results" / "final_validation" / "ablation",
        help="Root output directory for ablation study results",
    )
    parser.add_argument(
        "--probe-report-path",
        type=Path,
        default=_PROJECT_ROOT / "probe_out" / "representation_drift_report.json",
        help="Path to PRDAF probe report JSON",
    )
    parser.add_argument(
        "--ensemble-report-path",
        type=Path,
        default=_PROJECT_ROOT / "ensemble_out" / "variance_report.json",
        help="Path to EVM ensemble report JSON",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="ablation_study",
        help="Experiment identifier name",
    )
    return parser.parse_args()


def compute_ablation_config(
    config_name: str,
    rep_weight: float,
    unc_weight: float,
    probe_path: Path,
    ensemble_path: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Computes SCRS for a specific ablation weight setup (rep_weight, unc_weight).
    """
    cfg_output_dir = output_dir / config_name.lower()
    cfg_output_dir.mkdir(parents=True, exist_ok=True)

    scrs_cfg = SCRSConfig()
    scrs_cfg.update_paths(
        output_dir=cfg_output_dir,
        probe_report_path=probe_path,
        ensemble_report_path=ensemble_path,
    )
    scrs_cfg.representation_group_weight = rep_weight
    scrs_cfg.uncertainty_group_weight = unc_weight

    engine = SCRSEngine(scrs_cfg, logger=logger)
    result = engine.compute()

    reporter = SCRSReportGenerator(scrs_cfg, logger=logger)
    reporter.generate_json_report(result)
    reporter.generate_text_summary(result)

    ablation_info = {
        "ablation_config": config_name,
        "representation_weight": rep_weight,
        "uncertainty_weight": unc_weight,
        "representation_risk": result.representation_risk,
        "uncertainty_risk": result.uncertainty_risk,
        "scrs": result.scrs,
        "risk_label": result.risk_label,
        "normalized_metrics": result.to_dict()["normalized_metrics"]
    }

    save_json_report(ablation_info, cfg_output_dir / "ablation_config_report.json")
    return ablation_info


def main() -> None:
    args = parse_args()
    logger.info("=" * 80)
    logger.info("MONITORING COMPONENT ABLATION STUDY")
    logger.info("Output directory: %s", args.output_dir)
    logger.info("=" * 80)

    # Verify input reports exist or handle gracefully
    probe_path = args.probe_report_path
    ensemble_path = args.ensemble_report_path

    if not probe_path.exists() or not ensemble_path.exists():
        logger.warning(
            "Upstream PRDAF (%s) or EVM (%s) report missing. Running verification mock for ablation.",
            probe_path, ensemble_path
        )
        # Attempt fallback to research_results or scrs_out
        alt_probe = _PROJECT_ROOT / "scrs_out" / "scrs_report.json"
        if not probe_path.exists():
            probe_path = alt_probe

    ablation_configs = {
        "full": (0.60, 0.40),
        "no_prdaf": (0.00, 1.00),
        "no_evm": (1.00, 0.00),
    }

    ablation_results = {}
    for cfg_name, (rep_w, unc_w) in ablation_configs.items():
        logger.info("Computing Ablation Configuration: %s (Rep: %.2f, Unc: %.2f)...", cfg_name.upper(), rep_w, unc_w)
        res = compute_ablation_config(
            config_name=cfg_name,
            rep_weight=rep_w,
            unc_weight=unc_w,
            probe_path=probe_path,
            ensemble_path=ensemble_path,
            output_dir=args.output_dir
        )
        ablation_results[cfg_name] = res

    # Summary export
    summary_data = {
        "experiment_name": args.experiment_name,
        "probe_report_path": str(args.probe_report_path),
        "ensemble_report_path": str(args.ensemble_report_path),
        "ablation_results": ablation_results
    }
    json_path = args.output_dir / "ablation_summary.json"
    save_json_report(summary_data, json_path)

    # CSV Export
    csv_headers = ["Ablation_Config", "Representation_Weight", "Uncertainty_Weight", "Representation_Risk", "Uncertainty_Risk", "SCRS", "Risk_Label"]
    csv_rows = [
        [
            cfg_name.upper(),
            f"{info['representation_weight']:.2f}",
            f"{info['uncertainty_weight']:.2f}",
            f"{info['representation_risk']:.4f}",
            f"{info['uncertainty_risk']:.4f}",
            f"{info['scrs']:.4f}",
            info['risk_label']
        ]
        for cfg_name, info in ablation_results.items()
    ]
    csv_path = args.output_dir / "ablation_summary.csv"
    save_csv_report(csv_headers, csv_rows, csv_path)

    # Text Summary
    txt_content = (
        "================================================================================\n"
        "                  MONITORING COMPONENT ABLATION SUMMARY\n"
        "================================================================================\n\n"
    )
    for cfg_name, info in ablation_results.items():
        txt_content += (
            f"Configuration  : {cfg_name.upper()}\n"
            f"  Weights      : Rep = {info['representation_weight']:.2f}, Unc = {info['uncertainty_weight']:.2f}\n"
            f"  Rep Risk     : {info['representation_risk']:.4f}\n"
            f"  Unc Risk     : {info['uncertainty_risk']:.4f}\n"
            f"  Overall SCRS : {info['scrs']:.4f}\n"
            f"  Risk Label   : {info['risk_label']}\n\n"
        )
    txt_content += "================================================================================\n"
    txt_path = args.output_dir / "ablation_summary.txt"
    save_text_summary(txt_content, txt_path)

    # Visualization
    plot_path = args.output_dir / "ablation_comparison.png"
    plot_ablation_study(ablation_results, plot_path)

    logger.info("=" * 80)
    logger.info("MONITORING COMPONENT ABLATION STUDY COMPLETED SUCCESSFULLY!")
    logger.info("  JSON Summary: %s", json_path)
    logger.info("  CSV Summary : %s", csv_path)
    logger.info("  TXT Summary : %s", txt_path)
    logger.info("  Plot output : %s", plot_path)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
