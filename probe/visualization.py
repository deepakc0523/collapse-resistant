"""
================================================================================
probe/visualization.py
================================================================================

Generates publication-quality scientific charts showing representation drift,
attention alignment, logit divergence, and predictions.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Use 'Agg' non-interactive backend for server/Colab compatibility
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger("probe.visualization")

def generate_visualizations(report: Dict[str, Any], plots_dir: Path) -> None:
    """
    Generates and saves the required suite of diagnostic scientific plots.
    
    Plots:
        1. Layer-wise Representation Drift (Cosine & Euclidean)
        2. Attention Head Divergence Heatmap
        3. Embedding Static vs Dynamic Comparison
        4. Prediction Agreement & Overlap distribution
    """
    logger.info("Generating scientific visualizations under: %s", plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Set global plotting style
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "font.family": "serif",
        "figure.titlesize": 20,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.autolayout": True
    })

    # Retrieve metrics from report
    layer_wise = report["layer_wise_hidden_state_analysis"]
    embeddings = report["embedding_analysis"]
    logits = report["logit_analysis"]
    attention = report["attention_analysis"]
    
    layers = list(layer_wise.keys())
    layer_indices = [int(l.split("_")[1]) for l in layers]
    
    # -----------------------------------------------------------------
    # Plot 1: Layer-wise Representation Drift (Cosine Similarity)
    # -----------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    cosines = [layer_wise[l]["cosine_similarity"] for l in layers]
    euclideans = [layer_wise[l]["euclidean_distance"] for l in layers]
    
    color = "tab:blue"
    ax1.set_xlabel("Transformer Layer Index", fontweight="bold")
    ax1.set_ylabel("Cosine Similarity", color=color, fontweight="bold")
    line1 = ax1.plot(layer_indices, cosines, marker="o", color=color, linewidth=2.5, label="Cosine Similarity")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(-0.1, 1.1)
    
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Euclidean Distance", color=color, fontweight="bold")
    line2 = ax2.plot(layer_indices, euclideans, marker="s", color=color, linewidth=2.5, linestyle="--", label="Euclidean Distance")
    ax2.tick_params(axis="y", labelcolor=color)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center")
    
    plt.title("Representation Drift Profile across Transformer Layers", fontweight="bold", pad=15)
    plt.savefig(plots_dir / "representation_drift.png", dpi=150)
    plt.close()
    logger.info("Saved representation drift line plot.")

    # -----------------------------------------------------------------
    # Plot 2: Attention Head JSD Heatmap
    # -----------------------------------------------------------------
    head_jsd = np.array(attention["head_specific_jsd"]) # [layers, heads]
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        head_jsd,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=[f"H{h}" for h in range(head_jsd.shape[1])],
        yticklabels=[f"L{l}" for l in range(head_jsd.shape[0])],
        cbar_kws={"label": "Jensen-Shannon Divergence (JSD)"}
    )
    plt.title("Attention Distribution JSD Head-wise Mapping", fontweight="bold", pad=15)
    plt.xlabel("Attention Heads", fontweight="bold")
    plt.ylabel("Transformer Layers", fontweight="bold")
    plt.savefig(plots_dir / "attention_head_jsd.png", dpi=150)
    plt.close()
    logger.info("Saved attention alignment heatmap.")

    # -----------------------------------------------------------------
    # Plot 3: Embedding Similarity (Static vs Dynamic)
    # -----------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    
    embedding_types = ["Token (WTE) Static", "Token (WTE) Dynamic", "Pos (WPE) Static", "Pos (WPE) Dynamic"]
    cosine_values = [
        embeddings["static"]["token_embeddings_cosine_similarity"],
        embeddings["dynamic"]["token_embeddings_cosine_similarity_mean"],
        embeddings["static"]["position_embeddings_cosine_similarity"],
        embeddings["dynamic"]["position_embeddings_cosine_similarity_mean"]
    ]
    
    bars = plt.bar(embedding_types, cosine_values, color=["skyblue", "deepskyblue", "salmon", "darksalmon"], edgecolor="grey")
    plt.ylabel("Cosine Similarity", fontweight="bold")
    plt.title("Embedding Similarity: Static Parameters vs Contextual Activations", fontweight="bold", pad=15)
    plt.ylim(0.0, 1.1)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.0,
            height + 0.02,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )
        
    plt.xticks(rotation=15)
    plt.savefig(plots_dir / "embedding_similarity.png", dpi=150)
    plt.close()
    logger.info("Saved embedding similarity comparison chart.")

    # -----------------------------------------------------------------
    # Plot 4: Logit Divergence & Prediction Agreement Comparison
    # -----------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    
    metrics = ["Top-1 Agreement", "Top-5 Agreement", "Vocabulary Overlap (Jaccard)"]
    values = [
        logits["prediction_agreement_top1"] * 100,
        logits["prediction_agreement_top5"] * 100,
        logits.get("vocabulary_overlap_top50", 0.0) * 100
    ]
    
    colors = ["mediumseagreen", "seagreen", "darkcyan"]
    bars = plt.bar(metrics, values, color=colors, edgecolor="grey", width=0.6)
    plt.ylabel("Percentage (%)", fontweight="bold")
    plt.title("Output Logit Prediction & Token Overlap Agreement", fontweight="bold", pad=15)
    plt.ylim(0, 110)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.0,
            height + 2,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold"
        )
        
    plt.savefig(plots_dir / "prediction_agreement.png", dpi=150)
    plt.close()
    logger.info("Saved prediction agreement chart.")
    
    # -----------------------------------------------------------------
    # Plot 5: KL Divergence and JSD layer summary bar chart
    # -----------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    metrics_list = ["KL Divergence", "JS Divergence"]
    div_values = [logits["mean_kl_divergence"], logits["mean_js_divergence"]]
    
    plt.bar(metrics_list, div_values, color=["coral", "tomato"], width=0.4, edgecolor="grey")
    plt.ylabel("Divergence Value", fontweight="bold")
    plt.title("Logit Probability Divergence Metrics", fontweight="bold", pad=15)
    
    for idx, val in enumerate(div_values):
        plt.text(idx, val + 0.02 * max(div_values), f"{val:.5f}", ha="center", fontweight="bold")
        
    plt.savefig(plots_dir / "kl_js_divergence.png", dpi=150)
    plt.close()
    logger.info("Saved KL/JS divergence bar chart.")
    
    logger.info("[OK] All visualizations generated successfully.")
