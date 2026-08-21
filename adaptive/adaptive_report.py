"""
================================================================================
adaptive/adaptive_report.py
================================================================================

Report generator module for the Adaptive Threshold Engine (ATE).

Exports adaptive_out/adaptive_policy.json and adaptive_out/adaptive_summary.txt.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from adaptive.adaptive_config import AdaptiveConfig
from adaptive.scrs_loader import SCRSData
from adaptive.policy_engine import ATEPolicyResult
from adaptive.recommendation_engine import RecommendationReport
from adaptive.utils import get_adaptive_logger


class AdaptiveReportGenerator:
    """Generates JSON and TXT report artifacts for the Adaptive Threshold Engine."""

    def __init__(
        self,
        config: Optional[AdaptiveConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or AdaptiveConfig()
        self.logger = logger or get_adaptive_logger("adaptive.adaptive_report")

    def generate_reports(
        self,
        scrs_data: SCRSData,
        policy_result: ATEPolicyResult,
        rec_report: RecommendationReport,
    ) -> Tuple[Path, Path]:
        """
        Generates and writes both JSON and TXT reports to disk.

        Parameters
        ----------
        scrs_data : SCRSData
            Input SCRS metrics.
        policy_result : ATEPolicyResult
            Derived training policy result.
        rec_report : RecommendationReport
            Synthesized recommendations and scientific justifications.

        Returns
        -------
        Tuple[Path, Path]
            Paths to (adaptive_policy.json, adaptive_summary.txt).
        """
        json_path = self.write_json_report(scrs_data, policy_result, rec_report)
        txt_path = self.write_txt_summary(scrs_data, policy_result, rec_report)
        return json_path, txt_path

    def write_json_report(
        self,
        scrs_data: SCRSData,
        policy_result: ATEPolicyResult,
        rec_report: RecommendationReport,
    ) -> Path:
        """Writes machine-readable adaptive_policy.json artifact."""
        out_path = self.config.policy_json_path

        payload = {
            "metadata": {
                "title": "Adaptive Threshold Engine (ATE) Policy Report",
                "framework_version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "scrs_source_report": str(self.config.scrs_report_path),
            },
            "training_status": policy_result.training_status,
            "policy": policy_result.policy.to_dict(),
            "scrs_summary": policy_result.scrs_summary,
            "primary_risk_driver": rec_report.primary_risk_driver,
            "metric_influence": policy_result.metric_influence,
            "recommendations": rec_report.to_dict(),
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

        self.logger.info("Wrote adaptive policy JSON report: %s", out_path)
        return out_path

    def write_txt_summary(
        self,
        scrs_data: SCRSData,
        policy_result: ATEPolicyResult,
        rec_report: RecommendationReport,
    ) -> Path:
        """Writes human-readable adaptive_summary.txt artifact."""
        out_path = self.config.summary_txt_path
        policy = policy_result.policy

        lines = [
            "================================================================================",
            "          ADAPTIVE THRESHOLD ENGINE (ATE) POLICY REPORT                         ",
            "================================================================================",
            f" Timestamp           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f" Upstream SCRS Score : {scrs_data.overall_scrs:.4f} ({scrs_data.risk_label})",
            f" Training Status     : {policy_result.training_status}",
            f" Primary Risk Driver : {rec_report.primary_risk_driver}",
            "--------------------------------------------------------------------------------",
            " GENERATION-(N+1) RECOMMENDED POLICY                                            ",
            "--------------------------------------------------------------------------------",
            f"  • Synthetic Data Mix Ratio   : {policy.synthetic_ratio:.4f} ({policy.synthetic_ratio*100:.1f}%)",
            f"  • Anchor Data Mix Ratio      : {policy.anchor_ratio:.4f} ({policy.anchor_ratio*100:.1f}%)",
            f"  • Recommended Epochs         : {policy.recommended_epochs}",
            f"  • Recommended Learning Rate  : {policy.recommended_learning_rate:.6e}",
            f"  • Sampling Temperature       : {policy.sampling_temperature:.4f}",
            f"  • Max Generation Depth       : {policy.max_generation_depth}",
            f"  • Continue Recursive Train   : {policy.continue_recursive_training}",
            f"  • Risk Sensitivity Score (S) : {policy.risk_sensitivity_score:.4f}",
            "--------------------------------------------------------------------------------",
            " SCIENTIFIC RATIONALE & JUSTIFICATIONS                                         ",
            "--------------------------------------------------------------------------------",
        ]

        for i, just in enumerate(rec_report.justifications, 1):
            lines.append(f" [{i}] {just}")

        lines.extend([
            "--------------------------------------------------------------------------------",
            " RECOMMENDED MITIGATION ACTIONS                                                 ",
            "--------------------------------------------------------------------------------",
        ])

        for i, act in enumerate(rec_report.mitigation_actions, 1):
            lines.append(f" [{i}] {act}")

        lines.extend([
            "--------------------------------------------------------------------------------",
            " CURRICULUM CONTROLLER DIRECTIVES                                               ",
            "--------------------------------------------------------------------------------",
            json.dumps(rec_report.curriculum_instructions, indent=2),
            "================================================================================",
        ])

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        self.logger.info("Wrote adaptive summary text report: %s", out_path)
        return out_path
