"""
================================================================================
ensemble/visualization.py
================================================================================

Generates five publication-quality uncertainty visualisations for the EVM.

All figures use seaborn's 'whitegrid' theme with serif fonts and dpi=150
for publication readiness. The 'Agg' backend is used for server/headless
environments.

Figures generated
-----------------
1. entropy_distribution.png    — KDE + histogram of per-prompt normalised entropy
2. confidence_histogram.png    — Distribution of mean Top-1 confidence per prompt
3. variance_histogram.png      — Distribution of normalised probability variance
4. confidence_margin.png       — Distribution of Top-1 minus Top-2 confidence margins
5. mc_dropout_consistency.png  — Per-prompt MC Dropout consistency with mean line
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

# Use non-interactive backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger("ensemble.visualization")

# ---------------------------------------------------------------------------
# Global plot style
# ---------------------------------------------------------------------------
_STYLE_PARAMS = {
    "font.family":       "serif",
    "figure.titlesize":  20,
    "axes.titlesize":    16,
    "axes.labelsize":    14,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "figure.autolayout": True,
}

_PALETTE_BLUE    = "#2E86AB"
_PALETTE_TEAL    = "#17C3B2"
_PALETTE_VIOLET  = "#7B2D8B"
_PALETTE_ORANGE  = "#F18F01"
_PALETTE_GREEN   = "#44BBA4"
_MEAN_LINE_COLOR = "#E84855"   # Prominent red for mean reference lines


def _apply_style() -> None:
    """Apply global aesthetic style to all subsequent plots."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(_STYLE_PARAMS)


def _extract_per_prompt_field(report: Dict[str, Any], field: str) -> List[float]:
    """Safely extracts a named field from per_prompt_metrics, skipping NaN."""
    import math
    values = []
    for entry in report.get("per_prompt_metrics", []):
        v = entry.get(field)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            values.append(float(v))
    return values


# ---------------------------------------------------------------------------
# Figure 1 — Entropy Distribution
# ---------------------------------------------------------------------------

def _plot_entropy_distribution(report: Dict[str, Any], plots_dir: Path) -> None:
    """
    KDE + histogram of per-prompt normalised predictive entropy.

    Normalised entropy ∈ [0, 1]:
      - Concentrated near 0 → model is generally confident.
      - Concentrated near 1 → model is generally uncertain.
    """
    entropies = _extract_per_prompt_field(report, "predictive_entropy")
    if not entropies:
        logger.warning("No entropy data — skipping entropy_distribution plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(
        entropies,
        bins=20,
        kde=True,
        color=_PALETTE_BLUE,
        alpha=0.7,
        edgecolor="white",
        linewidth=0.6,
        ax=ax,
    )

    mean_val = float(np.mean(entropies))
    ax.axvline(mean_val, color=_MEAN_LINE_COLOR, linestyle="--", linewidth=2.0,
               label=f"Mean = {mean_val:.4f}")

    ax.set_xlabel("Normalised Predictive Entropy  [0 = confident | 1 = uncertain]",
                  fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Predictive Entropy Distribution across Prompts", fontweight="bold", pad=15)
    ax.legend(fontsize=12)

    _annotate_stats(ax, entropies)

    dest = plots_dir / "entropy_distribution.png"
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", dest)


# ---------------------------------------------------------------------------
# Figure 2 — Confidence Histogram
# ---------------------------------------------------------------------------

def _plot_confidence_histogram(report: Dict[str, Any], plots_dir: Path) -> None:
    """
    Histogram of per-prompt mean Top-1 confidence values.

    Top-1 confidence is the probability assigned to the model's argmax token
    at each position, averaged over the prompt's token sequence.
    """
    confidences = _extract_per_prompt_field(report, "top1_confidence")
    if not confidences:
        logger.warning("No confidence data — skipping confidence_histogram plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(
        confidences,
        bins=20,
        kde=True,
        color=_PALETTE_TEAL,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.6,
        ax=ax,
    )

    mean_val = float(np.mean(confidences))
    ax.axvline(mean_val, color=_MEAN_LINE_COLOR, linestyle="--", linewidth=2.0,
               label=f"Mean = {mean_val:.4f}")

    ax.set_xlabel("Top-1 Confidence  (max softmax probability per token, averaged)",
                  fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Top-1 Prediction Confidence Distribution", fontweight="bold", pad=15)
    ax.legend(fontsize=12)

    _annotate_stats(ax, confidences)

    dest = plots_dir / "confidence_histogram.png"
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", dest)


# ---------------------------------------------------------------------------
# Figure 3 — Variance Histogram
# ---------------------------------------------------------------------------

def _plot_variance_histogram(report: Dict[str, Any], plots_dir: Path) -> None:
    """
    Histogram of per-prompt normalised top-k probability variance.

    High variance implies a peaked distribution (confident predictions).
    Low variance implies a flat distribution (uncertain predictions).
    """
    variances = _extract_per_prompt_field(report, "probability_variance")
    if not variances:
        logger.warning("No variance data — skipping variance_histogram plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(
        variances,
        bins=20,
        kde=True,
        color=_PALETTE_VIOLET,
        alpha=0.70,
        edgecolor="white",
        linewidth=0.6,
        ax=ax,
    )

    mean_val = float(np.mean(variances))
    ax.axvline(mean_val, color=_MEAN_LINE_COLOR, linestyle="--", linewidth=2.0,
               label=f"Mean = {mean_val:.4f}")

    ax.set_xlabel("Normalised Probability Variance  (top-k distribution sharpness)",
                  fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Top-k Probability Variance Distribution", fontweight="bold", pad=15)
    ax.legend(fontsize=12)

    _annotate_stats(ax, variances)

    dest = plots_dir / "variance_histogram.png"
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", dest)


# ---------------------------------------------------------------------------
# Figure 4 — Confidence Margin Histogram
# ---------------------------------------------------------------------------

def _plot_confidence_margin(report: Dict[str, Any], plots_dir: Path) -> None:
    """
    Histogram of per-prompt Top-1 minus Top-2 confidence margins.

    Near-zero margin → model is confused between two tokens.
    Large margin → model strongly prefers one token.
    """
    margins = _extract_per_prompt_field(report, "confidence_margin")
    if not margins:
        logger.warning("No margin data — skipping confidence_margin plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(
        margins,
        bins=20,
        kde=True,
        color=_PALETTE_ORANGE,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.6,
        ax=ax,
    )

    mean_val = float(np.mean(margins))
    ax.axvline(mean_val, color=_MEAN_LINE_COLOR, linestyle="--", linewidth=2.0,
               label=f"Mean = {mean_val:.4f}")

    ax.set_xlabel("Confidence Margin  (p_top1 − p_top2, per token, averaged)",
                  fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Top-1 vs Top-2 Confidence Margin Distribution", fontweight="bold", pad=15)
    ax.legend(fontsize=12)

    _annotate_stats(ax, margins)

    dest = plots_dir / "confidence_margin.png"
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", dest)


# ---------------------------------------------------------------------------
# Figure 5 — Monte-Carlo Dropout Consistency
# ---------------------------------------------------------------------------

def _plot_mc_dropout_consistency(report: Dict[str, Any], plots_dir: Path) -> None:
    """
    Per-prompt MC Dropout consistency scatter/bar plot with a mean reference line.

    Consistency = fraction of MC passes agreeing with the majority-vote token,
    averaged over token positions. Value 1 = fully stable, 0 = fully unstable.
    """
    import math

    per_prompt = report.get("per_prompt_metrics", [])
    indices = []
    consistencies = []

    for entry in per_prompt:
        v = entry.get("mc_dropout_consistency")
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            indices.append(entry["prompt_index"])
            consistencies.append(float(v))

    if not consistencies:
        logger.warning("No MC dropout data — skipping mc_dropout_consistency plot.")
        return

    mean_val = float(np.mean(consistencies))

    fig, ax = plt.subplots(figsize=(max(10, len(indices) // 3), 6))

    # Scatter points coloured by consistency level
    scatter_colors = [
        _PALETTE_GREEN if c >= mean_val else _PALETTE_ORANGE
        for c in consistencies
    ]

    ax.bar(indices, consistencies, color=scatter_colors, alpha=0.75, edgecolor="white",
           linewidth=0.5, width=0.8)
    ax.axhline(mean_val, color=_MEAN_LINE_COLOR, linestyle="--", linewidth=2.0,
               label=f"Mean Consistency = {mean_val:.4f}")

    ax.set_xlabel("Prompt Index", fontweight="bold")
    ax.set_ylabel("MC Dropout Consistency  [0 = unstable | 1 = stable]", fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        "Monte-Carlo Dropout Prediction Consistency per Prompt",
        fontweight="bold",
        pad=15,
    )
    ax.legend(fontsize=12)

    # Annotate with overall statistics
    _annotate_stats(ax, consistencies, x_pos=0.02, y_pos=0.12)

    dest = plots_dir / "mc_dropout_consistency.png"
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", dest)


# ---------------------------------------------------------------------------
# Shared annotation helper
# ---------------------------------------------------------------------------

def _annotate_stats(
    ax: plt.Axes,
    values: List[float],
    x_pos: float = 0.98,
    y_pos: float = 0.95,
) -> None:
    """
    Adds a small statistics text box (mean, std, median, n) to the axes.
    """
    arr = np.array(values)
    stats_text = (
        f"n     = {len(arr)}\n"
        f"mean  = {arr.mean():.4f}\n"
        f"std   = {arr.std():.4f}\n"
        f"median= {np.median(arr):.4f}\n"
        f"min   = {arr.min():.4f}\n"
        f"max   = {arr.max():.4f}"
    )
    ax.text(
        x_pos, y_pos, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="grey"),
        fontfamily="monospace",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_visualizations(report: Dict[str, Any], plots_dir: Path) -> None:
    """
    Generates and saves all five EVM uncertainty visualisations.

    Args:
        report:    The master EVM report dictionary from variance_report.py.
        plots_dir: Directory where PNG files will be written.
    """
    logger.info("Generating EVM visualisations under: %s", plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    _apply_style()

    _plot_entropy_distribution(report, plots_dir)
    _plot_confidence_histogram(report, plots_dir)
    _plot_variance_histogram(report, plots_dir)
    _plot_confidence_margin(report, plots_dir)
    _plot_mc_dropout_consistency(report, plots_dir)

    logger.info("[OK] All 5 EVM visualisations generated successfully.")
