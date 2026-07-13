"""
================================================================================
probe/logit_analysis.py
================================================================================

Compares output distributions (logits) of the Anchor and Student models.
Measures divergence, prediction agreement, and vocabulary choice overlap.
"""

import torch
import logging
from typing import Dict, Any, List

from .similarity_metrics import (
    compute_kl_divergence,
    compute_js_divergence,
    compute_prediction_agreement,
    compute_vocabulary_overlap
)

logger = logging.getLogger("probe.logit_analysis")

def analyze_logits(
    anchor_features: Dict[str, Any],
    student_features: Dict[str, Any],
    top_k_vocab: int = 50
) -> Dict[str, Any]:
    """
    Computes comparative metrics on output logits of Anchor and Student.
    
    Args:
        anchor_features: Extracted Anchor features.
        student_features: Extracted Student features.
        top_k_vocab: The pool size for Jaccard index of vocabulary overlap.
        
    Returns:
        Dict containing average KL, JSD, top-1 agreement, top-5 agreement, and Jaccard overlap.
    """
    logger.info("Performing Logit and Prediction Divergence Analysis...")
    
    anchor_logits_list = anchor_features["logits"]
    student_logits_list = student_features["logits"]
    
    num_prompts = len(anchor_logits_list)
    
    kl_sum = 0.0
    jsd_sum = 0.0
    agree_top1_sum = 0.0
    agree_top5_sum = 0.0
    vocab_overlap_sum = 0.0
    
    total_tokens = 0
    
    for idx in range(num_prompts):
        a_logits = anchor_logits_list[idx]  # [seq_len, vocab_size]
        s_logits = student_logits_list[idx]  # [seq_len, vocab_size]
        
        seq_len = a_logits.size(0)
        total_tokens += seq_len
        
        # Calculate elementwise metrics
        kl = compute_kl_divergence(a_logits, s_logits)            # [seq_len]
        jsd = compute_js_divergence(a_logits, s_logits)            # [seq_len]
        agree_1 = compute_prediction_agreement(a_logits, s_logits, top_k=1)  # [seq_len]
        agree_5 = compute_prediction_agreement(a_logits, s_logits, top_k=5)  # [seq_len]
        overlap = compute_vocabulary_overlap(a_logits, s_logits, top_k=top_k_vocab) # [seq_len]
        
        kl_sum += kl.sum().item()
        jsd_sum += jsd.sum().item()
        agree_top1_sum += agree_1.sum().item()
        agree_top5_sum += agree_5.sum().item()
        vocab_overlap_sum += overlap.sum().item()
        
    # Average metrics per token
    metrics = {
        "mean_kl_divergence": kl_sum / total_tokens,
        "mean_js_divergence": jsd_sum / total_tokens,
        "prediction_agreement_top1": agree_top1_sum / total_tokens,
        "prediction_agreement_top5": agree_top5_sum / total_tokens,
        f"vocabulary_overlap_top{top_k_vocab}": vocab_overlap_sum / total_tokens,
    }
    
    logger.info(
        "[OK] Logit analysis completed. (KL Divergence: %.4f | Top-1 Agreement: %.2f%%)",
        metrics["mean_kl_divergence"],
        metrics["prediction_agreement_top1"] * 100
    )
    return metrics
