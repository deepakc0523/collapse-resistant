"""
================================================================================
experiments/visualization.py
================================================================================

Publication-quality visualization generator for experimental validation suite.
Generates research paper figures using Matplotlib with clean styling, custom color schemes,
and scientific formatting.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_publication_style() -> None:
    """Configures clean, publication-oriented Matplotlib styling."""
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#2C3E50",
        "axes.linewidth": 1.2,
        "grid.color": "#BDC3C7",
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "figure.dpi": 300,
    })


def plot_multiseed_summary(
    baseline_stats: Dict[str, float],
    adaptive_stats: Dict[str, float],
    output_path: Path
) -> Path:
    """
    Plots mean SCRS comparison with error bars (std) for Baseline vs Adaptive models.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    models = ["Student-2 Baseline", "Student-2 Adaptive"]
    means = [baseline_stats["mean"], adaptive_stats["mean"]]
    stds = [baseline_stats["std"], adaptive_stats["std"]]
    colors = ["#E74C3C", "#2ECC71"]

    bars = ax.bar(models, means, yerr=stds, capsize=8, color=colors, alpha=0.85, width=0.45, edgecolor="black")

    ax.set_ylabel("Synthetic Collapse Risk Score (SCRS)", fontsize=11, fontweight="bold")
    ax.set_title("Multi-Seed SCRS Comparison (Mean ± Std)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y")

    # Add numeric labels on top of bars
    for bar, mean_val, std_val in zip(bars, means, stds):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + std_val + 0.02,
            f"{mean_val:.4f}\n(±{std_val:.4f})",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_per_seed_comparison(
    seeds: List[int],
    baseline_scores: List[float],
    adaptive_scores: List[float],
    output_path: Path
) -> Path:
    """
    Plots per-seed SCRS trajectory comparison (Baseline vs Adaptive).
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    x_labels = [f"Seed {s}" for s in seeds]
    x_indices = range(len(seeds))

    ax.plot(x_indices, baseline_scores, marker="o", linewidth=2, color="#E74C3C", label="Student-2 Baseline")
    ax.plot(x_indices, adaptive_scores, marker="s", linewidth=2, color="#2ECC71", label="Student-2 Adaptive")

    ax.set_xticks(list(x_indices))
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel("Synthetic Collapse Risk Score (SCRS)", fontsize=11, fontweight="bold")
    ax.set_title("Per-Seed SCRS Evaluation: Baseline vs. Adaptive", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.legend(loc="upper right")

    for i in x_indices:
        ax.text(i, baseline_scores[i] + 0.015, f"{baseline_scores[i]:.3f}", ha="center", fontsize=8, color="#900C3F")
        ax.text(i, adaptive_scores[i] - 0.03, f"{adaptive_scores[i]:.3f}", ha="center", fontsize=8, color="#1E8449")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_ablation_study(
    ablation_results: Dict[str, Dict[str, Any]],
    output_path: Path
) -> Path:
    """
    Plots comparison across monitoring component ablation settings (Full, No-PRDAF, No-EVM).
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    configs = list(ablation_results.keys())
    labels = [c.upper().replace("_", "-") for c in configs]
    scores = [ablation_results[c]["scrs"] for c in configs]
    colors = ["#3498DB", "#E67E22", "#9B59B6"]

    bars = ax.bar(labels, scores, color=colors[:len(configs)], alpha=0.85, width=0.45, edgecolor="black")

    ax.set_ylabel("Synthetic Collapse Risk Score (SCRS)", fontsize=11, fontweight="bold")
    ax.set_title("Monitoring Component Ablation Study", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y")

    for bar, score in zip(bars, scores):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.02,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_weighting_sensitivity(
    sensitivity_data: Dict[str, List[Tuple[float, float]]],
    output_path: Path
) -> Path:
    """
    Plots SCRS score vs Representation Risk Weight for multiple models (Student-1, Baseline, Adaptive).
    
    sensitivity_data format: {model_name: [(rep_weight, scrs), ...]}
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    model_colors = {
        "Student-1": "#F39C12",
        "Student-2 Baseline": "#E74C3C",
        "Student-2 Adaptive": "#2ECC71",
    }

    for model_name, points in sensitivity_data.items():
        sorted_pts = sorted(points, key=lambda x: x[0])
        weights = [p[0] for p in sorted_pts]
        scores = [p[1] for p in sorted_pts]
        color = model_colors.get(model_name, "#34495E")
        
        ax.plot(weights, scores, marker="o", linewidth=2, label=model_name, color=color)

    ax.axvline(x=0.60, color="#7F8C8D", linestyle=":", label="Default (60:40)")

    ax.set_xlabel("Representation Risk Weight (W_rep)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Synthetic Collapse Risk Score (SCRS)", fontsize=11, fontweight="bold")
    ax.set_title("SCRS Sensitivity to Risk Weighting Configuration", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.legend(loc="best")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_layer_wise_cka(
    layer_cka_dict: Dict[str, float],
    output_path: Path
) -> Path:
    """
    Plots layer-wise linear CKA similarity curve across transformer layers.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    layers = list(layer_cka_dict.keys())
    scores = list(layer_cka_dict.values())

    ax.plot(layers, scores, marker="D", linewidth=2.5, color="#8E44AD", label="Linear CKA")
    ax.set_xlabel("Layer", fontsize=11, fontweight="bold")
    ax.set_ylabel("Centered Kernel Alignment (CKA)", fontsize=11, fontweight="bold")
    ax.set_title("Layer-Wise CKA Representation Validity Analysis", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True)
    ax.legend(loc="lower left")

    for idx, (layer, score) in enumerate(zip(layers, scores)):
        ax.text(idx, score + 0.02, f"{score:.4f}", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path
