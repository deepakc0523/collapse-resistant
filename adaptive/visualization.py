"""
================================================================================
adaptive/visualization.py
================================================================================

Publication-quality visualizer for the Adaptive Threshold Engine (ATE).

Generates 5 required visualization plots:
  1. policy_overview.png           (Baseline vs Derived Hyperparameters)
  2. metric_influence.png          (Individual Metric Influence Bar Chart)
  3. training_recommendations.png  (Ratio & Parameter Transfer Curves)
  4. recursive_pathway.png         (SCRS -> ATE Policy Decision Flow)
  5. policy_heatmap.png            (Hyperparameter Risk Intensity Matrix)
"""

import math
import logging
from pathlib import Path
from typing import Optional, List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adaptive.adaptive_config import AdaptiveConfig
from adaptive.scrs_loader import SCRSData
from adaptive.policy_engine import ATEPolicyResult
from adaptive.recommendation_engine import RecommendationReport
from adaptive.utils import get_adaptive_logger, sigmoid


class AdaptiveVisualizer:
    """Generates publication-quality visualizations for ATE results."""

    def __init__(
        self,
        config: Optional[AdaptiveConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or AdaptiveConfig()
        self.logger = logger or get_adaptive_logger("adaptive.visualization")
        self.plots_dir = self.config.plots_dir
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Matplotlib style
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def generate_all_plots(
        self,
        scrs_data: SCRSData,
        policy_result: ATEPolicyResult,
        rec_report: RecommendationReport,
    ) -> List[Path]:
        """Generates all 5 required visualization plots."""
        paths = []
        paths.append(self.plot_policy_overview(policy_result))
        paths.append(self.plot_metric_influence(policy_result))
        paths.append(self.plot_training_recommendations(policy_result))
        paths.append(self.plot_recursive_pathway(scrs_data, policy_result))
        paths.append(self.plot_policy_heatmap(policy_result))
        return paths

    def plot_policy_overview(self, policy_result: ATEPolicyResult) -> Path:
        """1. Baseline vs Derived Hyperparameters Overview."""
        out_path = self.plots_dir / "policy_overview.png"
        policy = policy_result.policy

        labels = [
            "Synthetic Ratio",
            "Anchor Ratio",
            "Epochs Ratio\n(Epochs/5)",
            "LR Scale\n(LR/1e-4)",
            "Temp Scale\n(Temp/1.0)",
            "Depth Scale\n(Depth/5)",
        ]

        # Derived relative values (normalized to [0, 1] for visual overview)
        derived_vals = [
            policy.synthetic_ratio,
            policy.anchor_ratio,
            policy.recommended_epochs / self.config.max_epochs,
            policy.recommended_learning_rate / self.config.max_learning_rate,
            policy.sampling_temperature / self.config.max_sampling_temperature,
            policy.max_generation_depth / self.config.max_generation_depth,
        ]

        # Nominal baseline normalized values
        baseline_vals = [
            0.70,
            0.30,
            self.config.base_epochs / self.config.max_epochs,
            self.config.base_learning_rate / self.config.max_learning_rate,
            self.config.base_sampling_temperature / self.config.max_sampling_temperature,
            3 / self.config.max_generation_depth,
        ]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, baseline_vals, width, label="Baseline (Nominal)", color="#1f77b4", alpha=0.7)
        rects2 = ax.bar(x + width/2, derived_vals, width, label=f"ATE Derived ({policy_result.training_status})", color="#ff7f0e", alpha=0.85)

        ax.set_ylabel("Normalized Scale [0, 1]", fontsize=12, fontweight="bold")
        ax.set_title(f"Policy Overview: Nominal Baseline vs. ATE Derived Policy ({policy_result.training_status})", fontsize=14, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=11, loc="upper right")

        # Value annotations
        for bar in rects2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f"{h:.2f}", ha='center', va='bottom', fontsize=9, fontweight="bold")

        plt.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        self.logger.info("Generated policy overview chart: %s", out_path)
        return out_path

    def plot_metric_influence(self, policy_result: ATEPolicyResult) -> Path:
        """2. Individual Metric Influence Bar Chart."""
        out_path = self.plots_dir / "metric_influence.png"
        influence = policy_result.metric_influence

        # Clean display names
        names = [k.replace("rep_", "Rep: ").replace("unc_", "Unc: ").replace("_", " ").title() for k in influence.keys()]
        vals = list(influence.values())

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#2b5c8f" if "rep_" in k else "#d95f02" for k in influence.keys()]

        bars = ax.barh(names, vals, color=colors, edgecolor="black", alpha=0.85)
        ax.set_xlabel("Metric Influence Contribution to SCRS Risk Policy", fontsize=12, fontweight="bold")
        ax.set_title("Metric Influence Breakdown on Adaptive Policy Decision", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlim(0, max(vals) * 1.25 if vals and max(vals) > 0 else 0.2)

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.002, bar.get_y() + bar.get_height()/2., f"{w:.4f}", va="center", ha="left", fontsize=9)

        plt.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        self.logger.info("Generated metric influence chart: %s", out_path)
        return out_path

    def plot_training_recommendations(self, policy_result: ATEPolicyResult) -> Path:
        """3. Continuous Ratio & Parameter Transfer Curves."""
        out_path = self.plots_dir / "training_recommendations.png"

        scrs_range = np.linspace(0.0, 1.0, 200)
        k = self.config.sigmoid_steepness
        x0 = self.config.sigmoid_midpoint

        syn_ratios = []
        anchor_ratios = []
        lrs = []

        for s in scrs_range:
            risk_s = sigmoid(s, k=k, x0=x0)
            syn = self.config.max_synthetic_ratio - risk_s * (self.config.max_synthetic_ratio - self.config.min_synthetic_ratio)
            anc = 1.0 - syn
            decay = (1.0 - risk_s) ** 2
            lr = self.config.min_learning_rate + (self.config.base_learning_rate - self.config.min_learning_rate) * decay
            syn_ratios.append(syn)
            anchor_ratios.append(anc)
            lrs.append(lr)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Subplot 1: Mix Ratios
        ax1.plot(scrs_range, syn_ratios, label="Synthetic Data Ratio", color="#1f77b4", linewidth=2.5)
        ax1.plot(scrs_range, anchor_ratios, label="Anchor Data Ratio", color="#ff7f0e", linewidth=2.5)
        current_scrs = policy_result.scrs_summary["overall_scrs"]
        ax1.axvline(current_scrs, color="red", linestyle="--", label=f"Current SCRS ({current_scrs:.4f})")
        ax1.set_xlabel("Overall SCRS Score", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Dataset Proportion", fontsize=11, fontweight="bold")
        ax1.set_title("Continuous Dataset Mix Ratios vs. SCRS Risk", fontsize=12, fontweight="bold")
        ax1.set_ylim(0, 1.0)
        ax1.legend(fontsize=10)

        # Subplot 2: Learning Rate Curve
        ax2.plot(scrs_range, lrs, label="Recommended LR", color="#2ca02c", linewidth=2.5)
        ax2.axvline(current_scrs, color="red", linestyle="--", label=f"Current SCRS ({current_scrs:.4f})")
        ax2.set_xlabel("Overall SCRS Score", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Learning Rate", fontsize=11, fontweight="bold")
        ax2.set_yscale("log")
        ax2.set_title("Smooth Learning Rate Scaling vs. SCRS Risk", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=10)

        plt.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        self.logger.info("Generated training recommendations curves: %s", out_path)
        return out_path

    def plot_recursive_pathway(
        self, scrs_data: SCRSData, policy_result: ATEPolicyResult
    ) -> Path:
        """4. SCRS -> ATE Policy Decision Flow Diagram."""
        out_path = self.plots_dir / "recursive_pathway.png"
        policy = policy_result.policy

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("off")

        # Diagram Boxes
        # Box 1: Upstream Signals (Probe + Ensemble)
        box_probe = dict(boxstyle="round,pad=0.5", fc="#e6f2ff", ec="#1f77b4", lw=2)
        ax.text(0.15, 0.75, f"Probe Drift Metrics\n(Rep Risk: {scrs_data.representation_risk:.4f})", ha="center", va="center", bbox=box_probe, fontsize=10, fontweight="bold")

        box_unc = dict(boxstyle="round,pad=0.5", fc="#fff0e6", ec="#ff7f0e", lw=2)
        ax.text(0.15, 0.25, f"Ensemble Variance Metrics\n(Unc Risk: {scrs_data.uncertainty_risk:.4f})", ha="center", va="center", bbox=box_unc, fontsize=10, fontweight="bold")

        # Box 2: SCRS Fusion
        box_scrs = dict(boxstyle="round,pad=0.6", fc="#e6ffe6", ec="#2ca02c", lw=2)
        ax.text(0.45, 0.50, f"SCRS Fusion Engine\nOverall Score: {scrs_data.overall_scrs:.4f}\nLabel: {scrs_data.risk_label}", ha="center", va="center", bbox=box_scrs, fontsize=11, fontweight="bold")

        # Box 3: ATE Decision Policy
        box_ate = dict(boxstyle="round,pad=0.6", fc="#f9f2ff", ec="#9467bd", lw=2.5)
        policy_str = (
            f"Adaptive Threshold Engine (ATE)\n"
            f"Status: {policy_result.training_status}\n"
            f"Synthetic Ratio: {policy.synthetic_ratio:.2f} | Anchor: {policy.anchor_ratio:.2f}\n"
            f"Epochs: {policy.recommended_epochs} | LR: {policy.recommended_learning_rate:.2e}\n"
            f"Temp: {policy.sampling_temperature:.2f} | Max Depth: {policy.max_generation_depth}"
        )
        ax.text(0.82, 0.50, policy_str, ha="center", va="center", bbox=box_ate, fontsize=10, fontweight="bold")

        # Arrows
        ax.annotate("", xy=(0.32, 0.60), xytext=(0.26, 0.70), arrowprops=dict(arrowstyle="->", lw=2, color="#1f77b4"))
        ax.annotate("", xy=(0.32, 0.40), xytext=(0.26, 0.30), arrowprops=dict(arrowstyle="->", lw=2, color="#ff7f0e"))
        ax.annotate("", xy=(0.67, 0.50), xytext=(0.58, 0.50), arrowprops=dict(arrowstyle="->", lw=2.5, color="#9467bd"))

        ax.set_title("Recursive Pathway: From SCRS Metrics to Generation-(N+1) Policy", fontsize=14, fontweight="bold", pad=20)

        plt.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        self.logger.info("Generated recursive pathway diagram: %s", out_path)
        return out_path

    def plot_policy_heatmap(self, policy_result: ATEPolicyResult) -> Path:
        """5. Hyperparameter Intensity Matrix Heatmap."""
        out_path = self.plots_dir / "policy_heatmap.png"
        policy = policy_result.policy

        metrics = ["Synthetic Ratio", "Anchor Ratio", "Epochs Scale", "LR Scale", "Temperature", "Depth Scale"]
        vals = [
            policy.synthetic_ratio,
            policy.anchor_ratio,
            policy.recommended_epochs / self.config.max_epochs,
            policy.recommended_learning_rate / self.config.max_learning_rate,
            policy.sampling_temperature / self.config.max_sampling_temperature,
            policy.max_generation_depth / self.config.max_generation_depth,
        ]

        matrix = np.array(vals).reshape(-1, 1)

        fig, ax = plt.subplots(figsize=(6, 7))
        im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")

        ax.set_yticks(np.arange(len(metrics)))
        ax.set_yticklabels(metrics, fontsize=11, fontweight="bold")
        ax.set_xticks([0])
        ax.set_xticklabels([f"Policy Intensity ({policy_result.training_status})"], fontsize=11, fontweight="bold")

        for i, val in enumerate(vals):
            color = "white" if val > 0.5 else "black"
            ax.text(0, i, f"{val:.4f}", ha="center", va="center", color=color, fontweight="bold", fontsize=11)

        cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.05)
        cbar.set_label("Normalized Intensity (0 to 1)", fontsize=11, fontweight="bold")

        ax.set_title("Policy Hyperparameter Heatmap", fontsize=14, fontweight="bold", pad=15)

        plt.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        self.logger.info("Generated policy heatmap: %s", out_path)
        return out_path
