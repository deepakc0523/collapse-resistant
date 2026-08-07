"""
================================================================================
scrs/scrs_report.py
================================================================================

Report generator module for SCRS.

Generates:
  1. scrs_out/scrs_report.json
  2. scrs_out/scrs_summary.txt
"""

import json
import logging
from pathlib import Path
from typing import Optional

from scrs.scrs_config import SCRSConfig
from scrs.scrs_engine import SCRSResult
from scrs.utils import get_scrs_logger


class SCRSReportGenerator:
    """Generates JSON and human-readable text summary reports for SCRS results."""

    def __init__(self, config: Optional[SCRSConfig] = None, logger: Optional[logging.Logger] = None) -> None:
        self.config = config or SCRSConfig()
        self.logger = logger or get_scrs_logger("scrs.scrs_report")

    def generate_json_report(self, result: SCRSResult, output_path: Optional[Path] = None) -> Path:
        """
        Generates scrs_report.json artifact.

        Parameters
        ----------
        result : SCRSResult
            Calculation result from SCRSEngine.
        output_path : Optional[Path]
            Destination path (defaults to config.report_json_path).

        Returns
        -------
        Path
            Path to the written JSON file.
        """
        dest_path = output_path or self.config.report_json_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        report_dict = result.to_dict()
        report_dict["metadata"] = {
            "title": "Synthetic Collapse Risk Score (SCRS) Report",
            "framework_version": "1.0",
            "description": "Unified mathematical fusion of representation drift (Probe) and prediction uncertainty (Ensemble).",
            "probe_report_source": str(self.config.probe_report_path),
            "ensemble_report_source": str(self.config.ensemble_report_path),
        }

        self.logger.info("Writing JSON report to %s", dest_path)
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=4)

        return dest_path

    def generate_text_summary(self, result: SCRSResult, output_path: Optional[Path] = None) -> Path:
        """
        Generates scrs_summary.txt artifact.

        Parameters
        ----------
        result : SCRSResult
            Calculation result from SCRSEngine.
        output_path : Optional[Path]
            Destination path (defaults to config.summary_txt_path).

        Returns
        -------
        Path
            Path to the written text file.
        """
        dest_path = output_path or self.config.summary_txt_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "=" * 80,
            "                   SYNTHETIC COLLAPSE RISK SCORE (SCRS) REPORT",
            "=" * 80,
            "",
            f"OVERALL SCRS SCORE  : {result.scrs:.4f} / 1.0000",
            f"RISK CATEGORY LABEL : {result.risk_label.upper()}",
            "",
            "-" * 80,
            "1. GROUP RISK BREAKDOWN",
            "-" * 80,
            f"  • Representation Risk (Weight: {result.group_weights['representation_group']*100:.1f}%): {result.representation_risk:.4f}",
            f"  • Uncertainty Risk    (Weight: {result.group_weights['uncertainty_group']*100:.1f}%): {result.uncertainty_risk:.4f}",
            "",
            "-" * 80,
            "2. REPRESENTATION RISK METRICS (Probe Upstream)",
            "-" * 80,
        ]

        for metric_name, risk_val in result.representation_metrics.items():
            w = result.representation_weights.get(metric_name, 0.0)
            contrib = result.metric_contributions.get(f"rep_{metric_name}", 0.0)
            lines.append(
                f"  - {metric_name:<24} | Risk: {risk_val:.4f} | Metric Weight: {w:.4f} | Total Contrib: {contrib:.4f}"
            )

        lines.extend([
            "",
            "-" * 80,
            "3. UNCERTAINTY RISK METRICS (Ensemble Upstream)",
            "-" * 80,
        ])

        for metric_name, risk_val in result.uncertainty_metrics.items():
            w = result.uncertainty_weights.get(metric_name, 0.0)
            contrib = result.metric_contributions.get(f"unc_{metric_name}", 0.0)
            lines.append(
                f"  - {metric_name:<24} | Risk: {risk_val:.4f} | Metric Weight: {w:.4f} | Total Contrib: {contrib:.4f}"
            )

        lines.extend([
            "",
            "-" * 80,
            "4. NORMALIZATION METADATA",
            "-" * 80,
            f"  • KL Divergence Bounds : Min = {self.config.kl_min:.2f}, Max = {self.config.kl_max:.2f}",
            f"  • Raw KL Divergence    : {result.probe_metrics.kl_divergence:.4f}",
            f"  • Raw JS Divergence    : {result.probe_metrics.js_divergence:.4f}",
            f"  • Raw Entropy          : {result.ensemble_metrics.predictive_entropy:.4f}",
            f"  • Raw MC Consistency   : {result.ensemble_metrics.mc_dropout_consistency:.4f}",
            "",
            "=" * 80,
            "                               END OF SCRS REPORT",
            "=" * 80,
        ])

        summary_text = "\n".join(lines)

        self.logger.info("Writing text summary report to %s", dest_path)
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        return dest_path
