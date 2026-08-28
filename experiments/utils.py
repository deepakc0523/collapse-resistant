"""
================================================================================
experiments/utils.py
================================================================================

Shared utility module for experimental validation framework.
Provides deterministic seed control, statistical comparison helpers (paired t-test,
Wilcoxon signed-rank test), pipeline execution runners, and artifact exporters.
"""

import os
import sys
import json
import csv
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch

# Configure logger
logger = logging.getLogger("experiments.utils")


def set_seed(seed: int) -> None:
    """
    Sets random seed across standard library, PyTorch, and environment for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_descriptive_stats(values: List[float]) -> Dict[str, float]:
    """
    Calculates mean, standard deviation, minimum, and maximum for a list of values.
    """
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        
    n = len(values)
    mean_val = sum(values) / n
    if n > 1:
        variance = sum((x - mean_val) ** 2 for x in values) / (n - 1)
        std_val = variance ** 0.5
    else:
        std_val = 0.0
        
    return {
        "mean": float(mean_val),
        "std": float(std_val),
        "min": float(min(values)),
        "max": float(max(values)),
        "count": n,
    }


def compute_paired_comparison_stats(
    baseline_scores: List[float],
    adaptive_scores: List[float]
) -> Dict[str, Any]:
    """
    Computes paired difference statistics, paired t-test, and Wilcoxon signed-rank test.
    Does NOT fabricate statistical significance; reports exact p-values and statistics.
    """
    if len(baseline_scores) != len(adaptive_scores):
        raise ValueError("Baseline and Adaptive score lists must have equal length for paired comparison.")
        
    n = len(baseline_scores)
    diffs = [a - b for a, b in zip(adaptive_scores, baseline_scores)]
    
    mean_diff = sum(diffs) / n if n > 0 else 0.0
    abs_improvement = -mean_diff  # Reduction in SCRS is improvement
    
    lower_count = sum(1 for d in diffs if d < 0)
    higher_count = sum(1 for d in diffs if d > 0)
    equal_count = sum(1 for d in diffs if d == 0)
    
    # Statistical tests
    paired_t_stat = None
    paired_t_pvalue = None
    wilcoxon_stat = None
    wilcoxon_pvalue = None
    test_method = "scipy.stats"
    
    try:
        from scipy import stats
        if n >= 2:
            t_res = stats.ttest_rel(adaptive_scores, baseline_scores)
            paired_t_stat = float(t_res.statistic) if not float('nan') in [t_res.statistic] else None
            paired_t_pvalue = float(t_res.pvalue) if not float('nan') in [t_res.pvalue] else None
            
            if any(d != 0 for d in diffs) and n >= 5:
                try:
                    w_res = stats.wilcoxon(adaptive_scores, baseline_scores)
                    wilcoxon_stat = float(w_res.statistic)
                    wilcoxon_pvalue = float(w_res.pvalue)
                except Exception:
                    pass
    except ImportError:
        test_method = "manual_fallback"
        if n >= 2:
            # Fallback manual paired t-test
            diff_mean = mean_diff
            var = sum((d - diff_mean) ** 2 for d in diffs) / (n - 1)
            std_err = (var / n) ** 0.5
            if std_err > 1e-9:
                paired_t_stat = diff_mean / std_err

    return {
        "num_pairs": n,
        "mean_difference_adaptive_minus_baseline": float(mean_diff),
        "mean_absolute_risk_reduction": float(abs_improvement),
        "per_seed_differences": [float(d) for d in diffs],
        "adaptive_lower_scrs_count": lower_count,
        "adaptive_higher_scrs_count": higher_count,
        "adaptive_equal_scrs_count": equal_count,
        "statistical_tests": {
            "method": test_method,
            "paired_t_test": {
                "statistic": paired_t_stat,
                "p_value": paired_t_pvalue,
            },
            "wilcoxon_signed_rank_test": {
                "statistic": wilcoxon_stat,
                "p_value": wilcoxon_pvalue,
            }
        }
    }


def save_json_report(data: Dict[str, Any], file_path: Path) -> None:
    """Saves dictionary data as formatted JSON."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_text_summary(text: str, file_path: Path) -> None:
    """Saves text summary file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)


def save_csv_report(headers: List[str], rows: List[List[Any]], file_path: Path) -> None:
    """Saves rows data as CSV."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
