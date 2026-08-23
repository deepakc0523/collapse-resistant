"""
================================================================================
scrs/compare_scrs.py
================================================================================

Comparative analysis script for Student-1, Student-2 Baseline, and Student-2 Adaptive
Synthetic Collapse Risk Scores (SCRS).

Usage:
    python -m scrs.compare_scrs
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent


def load_scrs_report(path: Path) -> Optional[Dict[str, Any]]:
    """Loads an SCRS JSON report file if it exists."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load report at {path}: {e}")
        return None


def run_scrs_comparison() -> None:
    """Compares SCRS reports across Student-1, Student-2 Baseline, and Student-2 Adaptive."""
    print("================================================================================")
    print("        RECURSIVE LEARNING COLLAPSE RISK COMPARISON (SCRS SUITE)")
    print("================================================================================\n")

    student1_path = _PROJECT_ROOT / "scrs_out" / "scrs_report.json"
    student2_base_path = _PROJECT_ROOT / "scrs_out" / "student2_baseline" / "scrs_report.json"
    student2_adap_path = _PROJECT_ROOT / "scrs_out" / "student2_adaptive" / "scrs_report.json"

    reports = {
        "Student-1 (Gen-1)": load_scrs_report(student1_path),
        "Student-2 Baseline (100% Syn)": load_scrs_report(student2_base_path),
        "Student-2 Adaptive (75:25 CC)": load_scrs_report(student2_adap_path),
    }

    print(f"{'Experiment Condition':<35} | {'SCRS Score':<12} | {'Risk Level':<12} | {'Representation Risk':<20} | {'Uncertainty Risk':<18}")
    print("-" * 105)

    comparison_data = []

    for name, r in reports.items():
        if r is None:
            print(f"{name:<35} | {'[NOT RUN]':<12} | {'N/A':<12} | {'N/A':<20} | {'N/A':<18}")
            comparison_data.append({
                "experiment": name,
                "status": "NOT_RUN",
            })
            continue

        scrs = r.get("scrs", 0.0)
        risk_label = r.get("risk_label", "Unknown").upper()
        rep_risk = r.get("representation_risk", 0.0)
        unc_risk = r.get("uncertainty_risk", 0.0)

        print(f"{name:<35} | {scrs:<12.4f} | {risk_label:<12} | {rep_risk:<20.4f} | {unc_risk:<18.4f}")
        comparison_data.append({
            "experiment": name,
            "status": "COMPLETED",
            "scrs": scrs,
            "risk_label": risk_label,
            "representation_risk": rep_risk,
            "uncertainty_risk": unc_risk,
        })

    print("-" * 105)

    # Export comparison output JSON
    output_json = _PROJECT_ROOT / "scrs_out" / "scrs_comparison.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=4)

    print(f"\nComparison JSON saved to: {output_json}\n")


if __name__ == "__main__":
    run_scrs_comparison()
