"""
================================================================================
ensemble/variance_report.py
================================================================================

Compiles EVM metric results into a structured JSON report and a human-readable
text summary.

JSON schema contract (SCRS-ready)
----------------------------------
The output JSON is designed so that SCRS can consume it directly in the next
module without changing field names. Key design decisions:

  - All normalized metrics are plain floats in [0, 1].
  - normalization_notes in metadata explains each metric's scale.
  - aggregate_metrics section provides means and std-devs for all six metrics.
  - per_prompt_metrics provides token-level granularity for prompt-level SCRS signals.
  - No collapse_risk, no threshold fields, no combined score.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger("ensemble.variance_report")

# ---------------------------------------------------------------------------
# Normalization metadata (embedded in report for SCRS consumption)
# ---------------------------------------------------------------------------
_NORMALIZATION_NOTES: Dict[str, str] = {
    "predictive_entropy": (
        "Shannon entropy H(p) = -sum_v p_v*log(p_v), normalized by log(vocab_size) "
        "to [0, 1]. Value 1 = maximum uncertainty (uniform distribution). "
        "Value 0 = maximum confidence (degenerate distribution)."
    ),
    "top1_confidence": (
        "Maximum probability max_v p_v. Native probability in [0, 1]. "
        "Value 1 = model places all mass on one token. "
        "Value 0 = model places essentially no mass on any single token."
    ),
    "top5_confidence_spread": (
        "Difference p_(1) - p_(5) between rank-1 and rank-5 probabilities. "
        "Both are probabilities so difference is in [0, 1]. "
        "Large value = mass concentrated at top. Small value = flat distribution."
    ),
    "probability_variance": (
        "Sample variance of top-k probabilities, normalized by the theoretical "
        "maximum variance (1/k)(1-1/k) for a k-item distribution. Range [0, 1]. "
        "High value = peaked/confident. Low value = flat/uncertain."
    ),
    "confidence_margin": (
        "Difference p_(1) - p_(2) between rank-1 and rank-2 probabilities. "
        "Both are probabilities so difference is in [0, 1]. "
        "Large margin = model strongly prefers one token. Near zero = confusion."
    ),
    "mc_dropout_consistency": (
        "Fraction of MC Dropout passes agreeing with the majority-vote prediction, "
        "averaged over token positions. Range [0, 1]. "
        "Value 1 = all passes agree (stable). Value 0 = all passes disagree (unstable)."
    ),
}


def compile_variance_report(
    metrics_result: Dict[str, Any],
    config_dict: Dict[str, Any],
    report_json_path: Path,
    summary_txt_path: Path,
) -> Dict[str, Any]:
    """
    Assembles the full EVM report from computed metrics and writes it to disk.

    Args:
        metrics_result:   Output of uncertainty_metrics.compute_all_metrics().
                          Must contain "per_prompt" and "aggregate" keys.
        config_dict:      Serialisable configuration snapshot for metadata.
        report_json_path: Destination path for the JSON report.
        summary_txt_path: Destination path for the human-readable summary.

    Returns:
        The complete master report dictionary (same structure as the JSON).
    """
    per_prompt: List[Dict[str, Any]] = metrics_result["per_prompt"]
    aggregate: Dict[str, float] = metrics_result["aggregate"]

    # -----------------------------------------------------------------------
    # Assemble master report
    # -----------------------------------------------------------------------
    master_report: Dict[str, Any] = {
        "metadata": {
            "title": "Ensemble Variance Monitor (EVM) — Prediction Uncertainty Report",
            "module": "evm",
            "framework_version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "description": (
                "Measures the Student model's prediction uncertainty using six "
                "calibrated metrics. All metrics are normalized to [0, 1]. "
                "This report is a direct upstream input to SCRS."
            ),
            "configurations": config_dict,
            "normalization_notes": _NORMALIZATION_NOTES,
        },
        "aggregate_metrics": {
            # Predictive Entropy
            "mean_predictive_entropy":     aggregate.get("mean_predictive_entropy"),
            "std_predictive_entropy":      aggregate.get("std_predictive_entropy"),
            # Top-1 Confidence
            "mean_top1_confidence":        aggregate.get("mean_top1_confidence"),
            "std_top1_confidence":         aggregate.get("std_top1_confidence"),
            # Top-5 Confidence Spread
            "mean_top5_confidence_spread": aggregate.get("mean_top5_confidence_spread"),
            "std_top5_confidence_spread":  aggregate.get("std_top5_confidence_spread"),
            # Probability Variance
            "mean_probability_variance":   aggregate.get("mean_probability_variance"),
            "std_probability_variance":    aggregate.get("std_probability_variance"),
            # Confidence Margin
            "mean_confidence_margin":      aggregate.get("mean_confidence_margin"),
            "std_confidence_margin":       aggregate.get("std_confidence_margin"),
            # MC Dropout Consistency
            "mean_mc_dropout_consistency": aggregate.get("mean_mc_dropout_consistency"),
            "std_mc_dropout_consistency":  aggregate.get("std_mc_dropout_consistency"),
        },
        "per_prompt_metrics": per_prompt,
    }

    # -----------------------------------------------------------------------
    # Write JSON report
    # -----------------------------------------------------------------------
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=4, default=_json_serializer)
    logger.info("Written JSON variance report to: %s", report_json_path)

    # -----------------------------------------------------------------------
    # Write human-readable text summary
    # -----------------------------------------------------------------------
    summary_txt = _build_summary_text(master_report, config_dict)
    summary_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_txt)
    logger.info("Written human-readable summary to: %s", summary_txt_path)

    return master_report


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _json_serializer(obj: Any) -> Any:
    """Fallback JSON serialiser for non-standard types (Path, float NaN)."""
    import math
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None  # JSON does not support NaN; use null
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable.")


def _fmt(value: Any, decimals: int = 6) -> str:
    """Format a numeric value, gracefully handling None and NaN."""
    import math
    if value is None:
        return "N/A"
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return f"{value:.{decimals}f}"


def _build_summary_text(report: Dict[str, Any], config_dict: Dict[str, Any]) -> str:
    """Builds the human-readable variance_summary.txt content."""
    agg = report["aggregate_metrics"]
    meta = report["metadata"]
    per_prompt = report["per_prompt_metrics"]

    lines = [
        "=" * 80,
        "ENSEMBLE VARIANCE MONITOR (EVM) — PREDICTION UNCERTAINTY REPORT",
        "=" * 80,
        f"Timestamp       : {meta['timestamp']}",
        f"Module          : {meta['module'].upper()}",
        f"Framework Ver.  : {meta['framework_version']}",
        "",
        "CONFIGURATION",
        "-" * 40,
        f"  Student Model : {config_dict.get('student_model_path', 'N/A')}",
        f"  Dataset Source: {config_dict.get('dataset_source', 'N/A')}",
        f"  Prompts       : {config_dict.get('max_prompts', 'N/A')}",
        f"  Batch Size    : {config_dict.get('batch_size', 'N/A')}",
        f"  MC Passes     : {config_dict.get('mc_dropout_passes', 'N/A')}",
        f"  Top-k         : {config_dict.get('top_k_confidence', 'N/A')}",
        f"  Device        : {config_dict.get('device', 'N/A')}",
        f"  Random Seed   : {config_dict.get('random_seed', 'N/A')}",
        "",
        "=" * 80,
        "AGGREGATE UNCERTAINTY METRICS  (all values normalised to [0, 1])",
        "=" * 80,
        "",
        "1. PREDICTIVE ENTROPY",
        "-" * 40,
        f"   Mean : {_fmt(agg['mean_predictive_entropy'])}",
        f"   Std  : {_fmt(agg['std_predictive_entropy'])}",
        "   Note : 0 = maximally confident | 1 = maximally uncertain",
        "",
        "2. TOP-1 CONFIDENCE",
        "-" * 40,
        f"   Mean : {_fmt(agg['mean_top1_confidence'])}",
        f"   Std  : {_fmt(agg['std_top1_confidence'])}",
        "   Note : Probability assigned to the most likely token",
        "",
        "3. TOP-5 CONFIDENCE SPREAD",
        "-" * 40,
        f"   Mean : {_fmt(agg['mean_top5_confidence_spread'])}",
        f"   Std  : {_fmt(agg['std_top5_confidence_spread'])}",
        "   Note : Difference between rank-1 and rank-5 probabilities",
        "",
        "4. PROBABILITY VARIANCE",
        "-" * 40,
        f"   Mean : {_fmt(agg['mean_probability_variance'])}",
        f"   Std  : {_fmt(agg['std_probability_variance'])}",
        "   Note : Normalised variance of top-k distribution sharpness",
        "",
        "5. CONFIDENCE MARGIN",
        "-" * 40,
        f"   Mean : {_fmt(agg['mean_confidence_margin'])}",
        f"   Std  : {_fmt(agg['std_confidence_margin'])}",
        "   Note : Gap between top-1 and top-2 probabilities (0 = confused)",
        "",
        "6. MONTE-CARLO DROPOUT CONSISTENCY",
        "-" * 40,
        f"   Mean : {_fmt(agg['mean_mc_dropout_consistency'])}",
        f"   Std  : {_fmt(agg['std_mc_dropout_consistency'])}",
        "   Note : Fraction of MC passes agreeing with majority vote",
        "",
        "=" * 80,
        f"PER-PROMPT BREAKDOWN  ({len(per_prompt)} prompts)",
        "=" * 80,
        f"{'IDX':>4}  {'ENTROPY':>9}  {'TOP1':>9}  {'SPREAD':>9}  "
        f"{'VAR':>9}  {'MARGIN':>9}  {'MC_CONS':>9}",
        "-" * 70,
    ]

    for r in per_prompt:
        lines.append(
            f"{r['prompt_index']:>4}  "
            f"{_fmt(r['predictive_entropy'], 6):>9}  "
            f"{_fmt(r['top1_confidence'], 6):>9}  "
            f"{_fmt(r['top5_confidence_spread'], 6):>9}  "
            f"{_fmt(r['probability_variance'], 6):>9}  "
            f"{_fmt(r['confidence_margin'], 6):>9}  "
            f"{_fmt(r['mc_dropout_consistency'], 6):>9}"
        )

    lines.extend([
        "",
        "=" * 80,
        "SCIENTIFIC INTERPRETATION",
        "=" * 80,
        "- Predictive Entropy and Top-1 Confidence are complementary: high entropy",
        "  corresponds to low confidence and vice versa.",
        "- Confidence Margin is a sharper signal than entropy — a near-zero margin",
        "  means the model is confused between exactly two tokens.",
        "- Probability Variance captures the shape of the top-k distribution —",
        "  high variance implies a peaked (confident) distribution.",
        "- MC Dropout Consistency measures epistemic uncertainty: low consistency",
        "  indicates the model's predictions are sensitive to dropout masking,",
        "  implying the model has not fully committed to a representation.",
        "=" * 80,
    ])

    return "\n".join(lines)
