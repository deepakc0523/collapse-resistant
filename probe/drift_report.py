"""
================================================================================
probe/drift_report.py
================================================================================

Aggregates all analysis results and compiles a structured scientific JSON report
along with a formatted human-readable summary text file.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import torch

from .similarity_metrics import compute_cosine_similarity, compute_euclidean_distance

logger = logging.getLogger("probe.drift_report")

def compute_layer_wise_similarities(
    anchor_features: Dict[str, Any],
    student_features: Dict[str, Any],
    num_layers: int = 6
) -> Dict[str, Dict[str, float]]:
    """
    Computes cosine similarity and Euclidean distance for every Transformer layer.
    
    Args:
        anchor_features: Extracted Anchor features.
        student_features: Extracted Student features.
        num_layers: Number of transformer blocks.
        
    Returns:
        Dict mapping layer names to their cosine similarity and Euclidean distance values.
    """
    logger.info("Computing Layer-wise Hidden State Similarities...")
    
    # hidden_states is a list (per prompt) of tuples. 
    # tuple index 0 is embedding, 1..6 are layers 0..5
    hs_anchor = anchor_features["hidden_states"]
    hs_student = student_features["hidden_states"]
    
    num_prompts = len(hs_anchor)
    layer_metrics = {}
    
    for l_idx in range(num_layers):
        cos_sum = 0.0
        euc_sum = 0.0
        total_tokens = 0
        
        for p_idx in range(num_prompts):
            # l_idx + 1 matches the output of the first block (index 0 is embedding)
            a_hs = hs_anchor[p_idx][l_idx + 1]  # [seq_len, hidden_dim]
            s_hs = hs_student[p_idx][l_idx + 1]  # [seq_len, hidden_dim]
            
            seq_len = a_hs.size(0)
            total_tokens += seq_len
            
            cos = compute_cosine_similarity(a_hs, s_hs, dim=-1) # [seq_len]
            euc = compute_euclidean_distance(a_hs, s_hs, dim=-1) # [seq_len]
            
            cos_sum += cos.sum().item()
            euc_sum += euc.sum().item()
            
        layer_metrics[f"layer_{l_idx}"] = {
            "cosine_similarity": cos_sum / total_tokens,
            "euclidean_distance": euc_sum / total_tokens
        }
        
    logger.info("[OK] Layer-wise similarities calculated.")
    return layer_metrics


def compile_drift_report(
    embedding_results: Dict[str, Any],
    attention_results: Dict[str, Any],
    logit_results: Dict[str, Any],
    layer_results: Dict[str, Any],
    config_dict: Dict[str, Any],
    report_json_path: Path,
    summary_txt_path: Path
) -> Dict[str, Any]:
    """
    Compiles all analysis structures into a master dictionary, writes it to JSON, 
    and generates a detailed scientific summary text file.
    """
    logger.info("Compiling representation drift reports...")
    
    master_report = {
        "metadata": {
            "title": "Anchor-Regularized Model Evolution (ARME) - Representation Drift Report",
            "framework_version": "1.0",
            "description": "Scientific analysis of student model representation collapse/drift compared to the frozen anchor.",
            "configurations": config_dict
        },
        "embedding_analysis": embedding_results,
        "attention_analysis": attention_results,
        "logit_analysis": logit_results,
        "layer_wise_hidden_state_analysis": layer_results
    }
    
    # Save JSON report
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=4)
    logger.info("Written JSON drift report to: %s", report_json_path)
    
    # Generate human-readable txt summary
    summary_lines = [
        "=" * 80,
        "ARME REPRESENTATION DRIFT ANALYSIS REPORT",
        "=" * 80,
        f"Anchor Model : {config_dict.get('anchor_model_path', 'checkpoints/anchor_model/frozen')}",
        f"Student Model: {config_dict.get('student_model_path', 'checkpoints/student_model/best')}",
        f"Prompt Source: {config_dict.get('dataset_source', 'clean_wikitext.txt')}",
        f"Processed    : {config_dict.get('max_prompts', 100)} prompts",
        "=" * 80,
        "",
        "1. EMBEDDING DRIFT ANALYSIS",
        "-" * 40,
        f"  Static Token Embedding (WTE) Cosine Similarity : {embedding_results['static']['token_embeddings_cosine_similarity']:.6f}",
        f"  Static Token Embedding (WTE) Euclidean Distance: {embedding_results['static']['token_embeddings_euclidean_distance']:.6f}",
        f"  Static Pos Embedding (WPE) Cosine Similarity   : {embedding_results['static']['position_embeddings_cosine_similarity']:.6f}",
        f"  Static Pos Embedding (WPE) Euclidean Distance  : {embedding_results['static']['position_embeddings_euclidean_distance']:.6f}",
        "",
        f"  Dynamic (Prompt Context) Token Embedding Cosine : {embedding_results['dynamic']['token_embeddings_cosine_similarity_mean']:.6f}",
        f"  Dynamic (Prompt Context) Pos Embedding Cosine   : {embedding_results['dynamic']['position_embeddings_cosine_similarity_mean']:.6f}",
        "",
        "2. LOGIT & PREDICTION DIVERGENCE",
        "-" * 40,
        f"  Mean KL Divergence (Anchor || Student)         : {logit_results['mean_kl_divergence']:.6f}",
        f"  Mean Jensen-Shannon Divergence (JSD)           : {logit_results['mean_js_divergence']:.6f}",
        f"  Top-1 Prediction Agreement                     : {logit_results['prediction_agreement_top1'] * 100:.2f}%",
        f"  Top-5 Prediction Agreement                     : {logit_results['prediction_agreement_top5'] * 100:.2f}%",
        f"  Vocabulary Overlap (Jaccard top-50)            : {logit_results.get('vocabulary_overlap_top50', 0.0) * 100:.2f}%",
        "",
        "3. LAYER-WISE REPRESENTATION ALIGNMENT",
        "-" * 40
    ]
    
    for layer_name, metrics in layer_results.items():
        summary_lines.append(
            f"  {layer_name.upper():<8} -> Cosine Similarity: {metrics['cosine_similarity']:.6f} | "
            f"Euclidean Distance: {metrics['euclidean_distance']:.6f}"
        )
        
    summary_lines.append("")
    summary_lines.append("4. LAYER-WISE ATTENTION MAP ALIGNMENT")
    summary_lines.append("-" * 40)
    for layer_name, cos_val in attention_results["layer_wise_attention_cosine_similarity"].items():
        jsd_val = attention_results["layer_wise_attention_jsd"][layer_name]
        summary_lines.append(
            f"  {layer_name.upper():<8} -> Attention Cosine Similarity: {cos_val:.6f} | "
            f"Attention JSD: {jsd_val:.6f}"
        )
        
    summary_lines.extend([
        "",
        "=" * 80,
        "DEEP LEARNING ANALYSIS / SCIENTIFIC COMMENTS",
        "=" * 80,
        "- Model collapse and representation drift are characterized by high Euclidean distance",
        "  and decreasing Cosine similarity, particularly in the later layers.",
        "- Top-1 Prediction Agreement indicates how closely the student's output policy mirrors",
        "  the frozen anchor. A high agreement is desirable in the early self-training iterations.",
        "- Jensen-Shannon Divergence is symmetric, offering a bounded metric [0, ln(2)] representing",
        "  the discrepancy between output policies. KL Divergence tracks informational loss.",
        "=" * 80
    ])
    
    summary_txt = "\n".join(summary_lines)
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_txt)
    logger.info("Written human-readable summary to: %s", summary_txt_path)
    
    return master_report
