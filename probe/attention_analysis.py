"""
================================================================================
probe/attention_analysis.py
================================================================================

Compares self-attention matrices and attention distributions between the Anchor 
and Student models. Computes layer-wise and head-wise alignment metrics.
"""

import torch
import logging
from typing import Dict, Any, List

logger = logging.getLogger("probe.attention_analysis")

def _kl_divergence_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Computes KL Divergence between two probability distributions directly."""
    return torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)), dim=-1)


def _js_divergence_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Computes Jensen-Shannon Divergence between two probability distributions."""
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence_probs(p, m, eps) + 0.5 * _kl_divergence_probs(q, m, eps)


def analyze_attention(
    anchor_features: Dict[str, Any],
    student_features: Dict[str, Any],
    num_layers: int = 6
) -> Dict[str, Any]:
    """
    Compares attention maps head-by-head and layer-by-layer across wikitext prompts.
    
    Args:
        anchor_features: Extracted Anchor features.
        student_features: Extracted Student features.
        num_layers: Number of transformer blocks.
        
    Returns:
        Dict containing layer-wise attention similarity and JSD metrics.
    """
    logger.info("Performing Attention Map Alignment Analysis...")
    
    # Anchor / Student attention lists (one per prompt)
    # Each list element is a tuple of length num_layers, where each element is [heads, seq, seq]
    att_anchor_list = anchor_features["attention_weights"]
    att_student_list = student_features["attention_weights"]
    
    num_prompts = len(att_anchor_list)
    
    # Initialise metrics dictionaries
    layer_jsd_mean = {layer_idx: 0.0 for layer_idx in range(num_layers)}
    layer_cos_mean = {layer_idx: 0.0 for layer_idx in range(num_layers)}
    
    # For head-specific analysis
    num_heads = att_anchor_list[0][0].size(0)
    head_jsd_matrix = torch.zeros(num_layers, num_heads)
    
    for p_idx in range(num_prompts):
        a_attn_tuple = att_anchor_list[p_idx]
        s_attn_tuple = att_student_list[p_idx]
        
        for l_idx in range(num_layers):
            a_att = a_attn_tuple[l_idx]  # [heads, seq_len, seq_len]
            s_att = s_attn_tuple[l_idx]  # [heads, seq_len, seq_len]
            
            # 1. Cosine similarity of the entire attention matrix (flattened)
            # Flatten across queries and keys: shape [heads, seq_len * seq_len]
            a_flat = a_att.flatten(1)
            s_flat = s_att.flatten(1)
            
            cos_sim = torch.cosine_similarity(a_flat, s_flat, dim=-1) # [heads]
            layer_cos_mean[l_idx] += cos_sim.mean().item()
            
            # 2. JS Divergence on the row-wise probability distributions (attention weights for each query)
            # Shape is [heads, seq_len, seq_len]. We treat the last dim (keys) as the distribution.
            jsd = _js_divergence_probs(a_att, s_att)  # [heads, seq_len]
            mean_jsd_per_head = jsd.mean(dim=-1)     # [heads]
            
            layer_jsd_mean[l_idx] += mean_jsd_per_head.mean().item()
            head_jsd_matrix[l_idx] += mean_jsd_per_head.cpu()

    # Normalise averages
    for l_idx in range(num_layers):
        layer_cos_mean[l_idx] /= num_prompts
        layer_jsd_mean[l_idx] /= num_prompts
        
    head_jsd_matrix /= num_prompts
    
    # Prepare result structure
    results = {
        "layer_wise_attention_cosine_similarity": {f"layer_{l}": val for l, val in layer_cos_mean.items()},
        "layer_wise_attention_jsd": {f"layer_{l}": val for l, val in layer_jsd_mean.items()},
        "head_specific_jsd": head_jsd_matrix.tolist()
    }
    
    logger.info("[OK] Attention alignment analysis completed.")
    return results
