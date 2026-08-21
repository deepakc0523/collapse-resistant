"""
================================================================================
curriculum/visualization.py
================================================================================

Publication-quality visualizer for the Curriculum Controller (CC).

Generates 5 required visualization plots:
  1. dataset_composition.png     (Anchor vs Synthetic Pie / Donut Chart)
  2. curriculum_progression.png  (Anchor / Synthetic Ratio Progression Curve)
  3. curriculum_schedule.png     (3-Stage Composition Bar Chart)
  4. sample_distribution.png     (Text Length Distribution Histogram)
  5. generation_flow.png        (End-to-End Dataset Construction Flowchart)
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from curriculum.curriculum_config import CurriculumConfig
from curriculum.policy_loader import ATEPolicyData
from curriculum.curriculum_scheduler import ScheduledCurriculum
from curriculum.dataset_loader import DatasetRecord
from curriculum.utils import get_curriculum_logger


class CurriculumVisualizer:
    """Generates publication-quality visualizations for Curriculum Controller results."""

    def __init__(
        self,
        config: Optional[CurriculumConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or CurriculumConfig()
        self.logger = logger or get_curriculum_logger("curriculum.visualization")
        self.plots_dir = self.config.plots_dir
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def generate_all_plots(
        self,
        policy_data: ATEPolicyData,
        scheduled: ScheduledCurriculum,
        records: List[DatasetRecord],
    ) -> List[Path]:
        """Generates all 5 required visualization plots."""
        paths = []
        paths.append(self.plot_dataset_composition(policy_data, records))
        paths.append(self.plot_curriculum_progression(scheduled))
        paths.append(self.plot_curriculum_schedule(scheduled))
        paths.append(self.plot_sample_distribution(records))
        paths.append(self.plot_generation_flow(policy_data, scheduled))
        return paths

    def plot_dataset_composition(
        self, policy_data: ATEPolicyData, records: List[DatasetRecord]
    ) -> Path:
        """1. Anchor vs Synthetic Donut Chart."""
        out_path = self.plots_dir / "dataset_composition.png"

        anc_count = sum(1 for r in records if r.source == "anchor")
        syn_count = sum(1 for r in records if r.source == "synthetic")

        sizes = [anc_count, syn_count]
        labels = [
            f"Anchor Dataset\n({anc_count} samples | {anc_count/len(records)*100:.1f}%)",
            f"Synthetic Dataset\n({syn_count} samples | {syn_count/len(records)*100:.1f}%)",
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
            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
        )

        plt.setp(autotexts, size=11, weight="bold", color="white")
        ax.set_title(f"Generation-2 Dataset Composition (Status: {policy_data.training_status})", fontsize=14, fontweight="bold", pad=20)

        plt.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        self.logger.info("Generated dataset composition plot: %s", out_path)
        return out_path

    def plot_curriculum_progression(self, scheduled: ScheduledCurriculum) -> Path:
        """2. Continuous Anchor/Synthetic Progression Curve across Sample Indexes."""
        out_path = self.plots_dir / "curriculum_progression.png"
        records = scheduled.ordered_records

        window_size = max(10, len(records) // 20)
        anchor_flags = [1 if r.source == "anchor" else 0 for r in records]

        # Moving average anchor ratio
        moving_avg_anc = np.convolve(anchor_flags, np.ones(window_size)/window_size, mode="same")
        moving_avg_syn = 1.0 - moving_avg_anc

        sample_idx = np.arange(len(records))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sample_idx, moving_avg_anc, label="Anchor Ratio (Moving Avg)", color="#1f77b4", linewidth=2.5)
        ax.plot(sample_idx, moving_avg_syn, label="Synthetic Ratio (Moving Avg)", color="#ff7f0e", linewidth=2.5)

        # Draw stage boundary vertical lines
        bounds = scheduled.stage_boundaries
        b1 = bounds.get("Stage_1_Foundation", [0, 0])[1]
        b2 = bounds.get("Stage_2_Transition", [0, 0])[1]

        ax.axvline(b1, color="black", linestyle="--", alpha=0.7, label="Stage 1 / Stage 2 Boundary")
        ax.axvline(b2, color="purple", linestyle="--", alpha=0.7, label="Stage 2 / Stage 3 Boundary")

        ax.set_xlabel("Sample Index in Generation-2 Dataset", fontsize=11, fontweight="bold")
        ax.set_ylabel("Proportion Ratio", fontsize=11, fontweight="bold")
        ax.set_title("Progressive Curriculum Exposure Curve (Foundation -> Target Mix)", fontsize=14, fontweight="bold", pad=15)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10, loc="center right")

        plt.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        self.logger.info("Generated curriculum progression curve: %s", out_path)
        return out_path

    def plot_curriculum_schedule(self, scheduled: ScheduledCurriculum) -> Path:
        """3. 3-Stage Composition Bar Chart."""
        out_path = self.plots_dir / "curriculum_schedule.png"
        comp = scheduled.stage_compositions

        stages = ["Stage 1\n(Foundation)", "Stage 2\n(Transition)", "Stage 3\n(Advanced Exposure)"]
        anc_ratios = [
            comp.get("Stage_1_Foundation", {}).get("anchor_ratio", 1.0),
            comp.get("Stage_2_Transition", {}).get("anchor_ratio", 0.8),
            comp.get("Stage_3_Advanced", {}).get("anchor_ratio", 0.5),
        ]
        syn_ratios = [
            comp.get("Stage_1_Foundation", {}).get("synthetic_ratio", 0.0),
            comp.get("Stage_2_Transition", {}).get("synthetic_ratio", 0.2),
            comp.get("Stage_3_Advanced", {}).get("synthetic_ratio", 0.5),
        ]

        x = np.arange(len(stages))
        width = 0.45

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.bar(x, anc_ratios, width, label="Anchor Ratio", color="#1f77b4", alpha=0.85)
        ax.bar(x, syn_ratios, width, bottom=anc_ratios, label="Synthetic Ratio", color="#ff7f0e", alpha=0.85)

        ax.set_ylabel("Stage Composition Fraction", fontsize=11, fontweight="bold")
        ax.set_title("3-Stage Progressive Curriculum Composition Breakdown", fontsize=14, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(stages, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10, loc="upper right")

        # Value labels
        for i in range(len(stages)):
            ax.text(i, anc_ratios[i]/2, f"Anc: {anc_ratios[i]:.2f}", ha="center", va="center", color="white", fontweight="bold")
            if syn_ratios[i] > 0.05:
                ax.text(i, anc_ratios[i] + syn_ratios[i]/2, f"Syn: {syn_ratios[i]:.2f}", ha="center", va="center", color="white", fontweight="bold")

        plt.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        self.logger.info("Generated curriculum schedule chart: %s", out_path)
        return out_path

    def plot_sample_distribution(self, records: List[DatasetRecord]) -> Path:
        """4. Text Length Distribution Histogram."""
        out_path = self.plots_dir / "sample_distribution.png"

        anc_lens = [len(r.text) for r in records if r.source == "anchor"]
        syn_lens = [len(r.text) for r in records if r.source == "synthetic"]

        fig, ax = plt.subplots(figsize=(9, 5))
        if anc_lens:
            ax.hist(anc_lens, bins=25, alpha=0.6, label="Human Anchor Records", color="#1f77b4", edgecolor="black")
        if syn_lens:
            ax.hist(syn_lens, bins=25, alpha=0.6, label="Synthetic Generation-1 Records", color="#ff7f0e", edgecolor="black")

        ax.set_xlabel("Sample Record Text Length (Characters)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Record Count", fontsize=11, fontweight="bold")
        ax.set_title("Sample Character Length Distribution (Anchor vs. Synthetic)", fontsize=14, fontweight="bold", pad=15)
        ax.legend(fontsize=10)

        plt.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        self.logger.info("Generated sample distribution plot: %s", out_path)
        return out_path

    def plot_generation_flow(
        self, policy_data: ATEPolicyData, scheduled: ScheduledCurriculum
    ) -> Path:
        """5. End-to-End Generation-2 Construction Flowchart."""
        out_path = self.plots_dir / "generation_flow.png"

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.axis("off")

        # Flowchart boxes
        box_input1 = dict(boxstyle="round,pad=0.5", fc="#e6f2ff", ec="#1f77b4", lw=2)
        ax.text(0.15, 0.75, "Canonical Human Anchor\nDataset (processed/)", ha="center", va="center", bbox=box_input1, fontsize=10, fontweight="bold")

        box_input2 = dict(boxstyle="round,pad=0.5", fc="#fff0e6", ec="#ff7f0e", lw=2)
        ax.text(0.15, 0.25, "Synthetic Data\nGeneration-1", ha="center", va="center", bbox=box_input2, fontsize=10, fontweight="bold")

        box_ate = dict(boxstyle="round,pad=0.5", fc="#f9f2ff", ec="#9467bd", lw=2)
        ax.text(0.45, 0.85, f"ATE Policy Input\nSyn Ratio: {policy_data.synthetic_ratio:.2f}\nAnc Ratio: {policy_data.anchor_ratio:.2f}", ha="center", va="center", bbox=box_ate, fontsize=9, fontweight="bold")

        box_cc = dict(boxstyle="round,pad=0.6", fc="#e6ffe6", ec="#2ca02c", lw=2.5)
        ax.text(0.45, 0.50, f"Curriculum Controller (CC)\n• Deterministic Mixing\n• 3-Stage Scheduling\n• Data Integrity Validation", ha="center", va="center", bbox=box_cc, fontsize=10, fontweight="bold")

        box_out = dict(boxstyle="round,pad=0.6", fc="#ffffcc", ec="#d62728", lw=2.5)
        ax.text(0.82, 0.50, f"Generation-2 Dataset\n(curriculum_out/generation_2/)\n• train.jsonl\n• validation.jsonl\n• metadata.json", ha="center", va="center", bbox=box_out, fontsize=10, fontweight="bold")

        # Connectors
        ax.annotate("", xy=(0.32, 0.60), xytext=(0.26, 0.70), arrowprops=dict(arrowstyle="->", lw=2, color="#1f77b4"))
        ax.annotate("", xy=(0.32, 0.40), xytext=(0.26, 0.30), arrowprops=dict(arrowstyle="->", lw=2, color="#ff7f0e"))
        ax.annotate("", xy=(0.45, 0.65), xytext=(0.45, 0.75), arrowprops=dict(arrowstyle="->", lw=2, color="#9467bd"))
        ax.annotate("", xy=(0.68, 0.50), xytext=(0.58, 0.50), arrowprops=dict(arrowstyle="->", lw=2.5, color="#2ca02c"))

        ax.set_title("Generation-2 Adaptive Dataset Construction Pipeline", fontsize=14, fontweight="bold", pad=20)

        plt.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        self.logger.info("Generated generation flow diagram: %s", out_path)
        return out_path
