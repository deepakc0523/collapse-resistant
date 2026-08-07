"""
================================================================================
scrs/visualization.py
================================================================================

Publication-quality visualization generator for SCRS.

Generates:
  1. metric_contribution.png               (Bar chart)
  2. representation_vs_uncertainty_radar.png (Radar chart)
  3. scrs_gauge.png                        (Risk gauge)
  4. risk_pie_chart.png                    (Pie / Donut chart)
  5. normalized_heatmap.png                (Heatmap)
"""

import math
import logging
from pathlib import Path
from typing import Optional, List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scrs.scrs_config import SCRSConfig
from scrs.scrs_engine import SCRSResult
from scrs.utils import get_scrs_logger


class SCRSVisualizer:
    """Generates publication-quality visualizations for SCRS results."""

    def __init__(self, config: Optional[SCRSConfig] = None, logger: Optional[logging.Logger] = None) -> None:
        self.config = config or SCRSConfig()
        self.logger = logger or get_scrs_logger("scrs.visualization")
        self.plots_dir = self.config.plots_dir
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Style preferences
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def generate_all_plots(self, result: SCRSResult) -> List[Path]:
        """Generates all 5 required visualization plots."""
        paths = []
        paths.append(self.plot_metric_contributions(result))
        paths.append(self.plot_radar_chart(result))
        paths.append(self.plot_scrs_gauge(result))
        paths.append(self.plot_risk_pie_chart(result))
        paths.append(self.plot_normalized_heatmap(result))
        return paths

    def plot_metric_contributions(self, result: SCRSResult) -> Path:
        """1. Metric Contribution Bar Chart."""
        out_path = self.plots_dir / "metric_contribution.png"
        
        contribs = result.metric_contributions
        labels = [k.replace("rep_", "Rep: ").replace("unc_", "Unc: ").replace("_", " ").title() for k in contribs.keys()]
        values = list(contribs.values())

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#1f77b4" if "rep_" in k else "#ff7f0e" for k in contribs.keys()]

        bars = ax.barh(labels, values, color=colors, edgecolor="black", alpha=0.85)
        ax.set_xlabel("Contribution to Total SCRS", fontsize=12, fontweight="bold")
        ax.set_title("Individual Metric Contributions to SCRS", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlim(0, max(values) * 1.25 if values and max(values) > 0 else 0.2)

        # Add value labels
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", ha="left", fontsize=9)

        plt.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        self.logger.info("Generated metric contribution bar chart: %s", out_path)
        return out_path

    def plot_radar_chart(self, result: SCRSResult) -> Path:
        """2. Representation vs Uncertainty Radar Chart."""
        out_path = self.plots_dir / "representation_vs_uncertainty_radar.png"

        # Prepare metrics for 6 dimensions (Rep vs Unc metrics aligned)
        rep_vals = list(result.representation_metrics.values())
        unc_vals = list(result.uncertainty_metrics.values())
        categories = ["Metric 1", "Metric 2", "Metric 3", "Metric 4", "Metric 5", "Metric 6"]

        N = len(categories)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]

        rep_vals += rep_vals[:1]
        unc_vals += unc_vals[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], categories, fontsize=11, fontweight="bold")

        ax.plot(angles, rep_vals, linewidth=2, linestyle="solid", label="Representation Risk (Probe)", color="#1f77b4")
        ax.fill(angles, rep_vals, color="#1f77b4", alpha=0.25)

        ax.plot(angles, unc_vals, linewidth=2, linestyle="solid", label="Uncertainty Risk (Ensemble)", color="#ff7f0e")
        ax.fill(angles, unc_vals, color="#ff7f0e", alpha=0.25)

        ax.set_ylim(0, 1.0)
        plt.title("Representation Risk vs. Uncertainty Risk Comparison", fontsize=14, fontweight="bold", pad=25)
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

        plt.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        self.logger.info("Generated radar chart: %s", out_path)
        return out_path

    def plot_scrs_gauge(self, result: SCRSResult) -> Path:
        """3. SCRS Risk Gauge Chart."""
        out_path = self.plots_dir / "scrs_gauge.png"

        fig, ax = plt.subplots(figsize=(8, 5), subplot_kw=dict(polar=True))
        ax.set_theta_offset(math.pi)
        ax.set_theta_direction(-1)

        # Gauge arc from 0 to pi (180 deg)
        # Levels: Very Low (green), Low (light green), Moderate (yellow), High (orange), Critical (red)
        gauge_colors = ["#2ca02c", "#bcbd22", "#e377c2", "#ff7f0e", "#d62728"]
        boundaries = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        for i in range(len(gauge_colors)):
            start_angle = boundaries[i] * math.pi
            end_angle = boundaries[i + 1] * math.pi
            ax.bar(
                x=(start_angle + end_angle) / 2,
                height=0.3,
                width=(end_angle - start_angle),
                bottom=0.7,
                color=gauge_colors[i],
                alpha=0.8,
                edgecolor="white",
            )

        # Gauge Needle
        needle_angle = result.scrs * math.pi
        ax.annotate(
            "",
            xy=(needle_angle, 0.95),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", lw=3, color="black"),
        )

        ax.set_ylim(0, 1.0)
        ax.set_yticks([])
        ax.set_xticks([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi])
        ax.set_xticklabels(["0.0 (Good)", "0.25", "0.50", "0.75", "1.0 (Bad)"], fontsize=10)

        plt.title(
            f"SCRS Gauge: {result.scrs:.4f} ({result.risk_label})",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        self.logger.info("Generated gauge chart: %s", out_path)
        return out_path

    def plot_risk_pie_chart(self, result: SCRSResult) -> Path:
        """4. Risk Contribution Pie Chart (Representation vs Uncertainty)."""
        out_path = self.plots_dir / "risk_pie_chart.png"

        rep_contrib = result.representation_risk * result.group_weights["representation_group"]
        unc_contrib = result.uncertainty_risk * result.group_weights["uncertainty_group"]

        sizes = [rep_contrib, unc_contrib]
        labels = [
            f"Representation Risk\n({rep_contrib:.4f})",
            f"Uncertainty Risk\n({unc_contrib:.4f})",
        ]
        colors = ["#1f77b4", "#ff7f0e"]

        fig, ax = plt.subplots(figsize=(7, 7))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            textprops=dict(fontsize=11, fontweight="bold"),
            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),  # Donut style
        )

        plt.setp(autotexts, size=11, weight="bold", color="white")
        ax.set_title("Group Contributions to Total SCRS", fontsize=14, fontweight="bold", pad=20)

        plt.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        self.logger.info("Generated pie chart: %s", out_path)
        return out_path

    def plot_normalized_heatmap(self, result: SCRSResult) -> Path:
        """5. Normalized Metric Risk Heatmap."""
        out_path = self.plots_dir / "normalized_heatmap.png"

        rep_metrics = result.representation_metrics
        unc_metrics = result.uncertainty_metrics

        all_names = list(rep_metrics.keys()) + list(unc_metrics.keys())
        all_vals = list(rep_metrics.values()) + list(unc_metrics.values())

        clean_names = [n.replace("_", " ").title() for n in all_names]
        matrix = np.array(all_vals).reshape(-1, 1)

        fig, ax = plt.subplots(figsize=(6, 8))
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")

        ax.set_yticks(np.arange(len(clean_names)))
        ax.set_yticklabels(clean_names, fontsize=10, fontweight="bold")
        ax.set_xticks([0])
        ax.set_xticklabels(["Risk Scale [0, 1]"], fontsize=11, fontweight="bold")

        # Add text annotations
        for i in range(len(all_vals)):
            val = all_vals[i]
            color = "white" if val > 0.6 else "black"
            ax.text(0, i, f"{val:.4f}", ha="center", va="center", color=color, fontweight="bold", fontsize=10)

        cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.05)
        cbar.set_label("Normalized Risk (0=Good, 1=Bad)", fontsize=11, fontweight="bold")

        ax.set_title("Normalized Metric Risk Heatmap", fontsize=14, fontweight="bold", pad=15)

        plt.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        self.logger.info("Generated heatmap: %s", out_path)
        return out_path
