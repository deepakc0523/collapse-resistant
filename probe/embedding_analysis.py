"""
================================================================================
probe/embedding_analysis.py
================================================================================

Analyzes representation drift in the input embedding layers (Token and Positional).
Computes static matrix correlation and dynamic prompt-context similarity.
"""

import torch
import logging
from typing import Dict, Any, List
from transformers import PreTrainedModel

from .similarity_metrics import compute_cosine_similarity, compute_euclidean_distance

logger = logging.getLogger("probe.embedding_analysis")

def analyze_embeddings(
    anchor_model: PreTrainedModel,
    student_model: PreTrainedModel,
    anchor_features: Dict[str, Any],
    student_features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compares token and position embeddings statically (weights) and dynamically (prompts).
    
    Args:
        anchor_model: The reference Anchor model.
        student_model: The evaluated Student model.
        anchor_features: Extracted features from the Anchor.
        student_features: Extracted features from the Student.
        
    Returns:
        Dict containing global and contextual embedding metrics.
    """
    logger.info("Performing Embedding Drift Analysis...")
    
    metrics = {}
    
    # -------------------------------------------------------------
    # 1. Global Parameter Matrix Analysis
    # -------------------------------------------------------------
    with torch.no_grad():
        wte_anchor = anchor_model.transformer.wte.weight.cpu()
        wte_student = student_model.transformer.wte.weight.cpu()
        
        wpe_anchor = anchor_model.transformer.wpe.weight.cpu()
        wpe_student = student_model.transformer.wpe.weight.cpu()
        
        # Token Embedding (WTE) Matrix stats
        wte_cos = compute_cosine_similarity(wte_anchor, wte_student, dim=-1).mean().item()
        wte_euc = compute_euclidean_distance(wte_anchor, wte_student, dim=-1).mean().item()
        
        # Position Embedding (WPE) Matrix stats
        wpe_cos = compute_cosine_similarity(wpe_anchor, wpe_student, dim=-1).mean().item()
        wpe_euc = compute_euclidean_distance(wpe_anchor, wpe_student, dim=-1).mean().item()
        
    metrics["static"] = {
        "token_embeddings_cosine_similarity": wte_cos,
        "token_embeddings_euclidean_distance": wte_euc,
        "position_embeddings_cosine_similarity": wpe_cos,
        "position_embeddings_euclidean_distance": wpe_euc,
    }
    
    # -------------------------------------------------------------
    # 2. Contextual Activation Analysis (Dynamic)
    # -------------------------------------------------------------
    anchor_tokens_emb = anchor_features["token_embeddings"]
    student_tokens_emb = student_features["token_embeddings"]
    
    anchor_pos_emb = anchor_features["position_embeddings"]
    student_pos_emb = student_features["position_embeddings"]
    
    num_prompts = len(anchor_tokens_emb)
    
    dynamic_wte_cos = []
    dynamic_wte_euc = []
    dynamic_wpe_cos = []
    dynamic_wpe_euc = []
    
    for idx in range(num_prompts):
        a_tok = anchor_tokens_emb[idx]
        s_tok = student_tokens_emb[idx]
        
        a_pos = anchor_pos_emb[idx]
        s_pos = student_pos_emb[idx]
        
        # Calculate cosine and Euclidean across the tokens of the prompt
        tok_cos = compute_cosine_similarity(a_tok, s_tok, dim=-1).mean().item()
        tok_euc = compute_euclidean_distance(a_tok, s_tok, dim=-1).mean().item()
        
        pos_cos = compute_cosine_similarity(a_pos, s_pos, dim=-1).mean().item()
        pos_euc = compute_euclidean_distance(a_pos, s_pos, dim=-1).mean().item()
        
        dynamic_wte_cos.append(tok_cos)
        dynamic_wte_euc.append(tok_euc)
        dynamic_wpe_cos.append(pos_cos)
        dynamic_wpe_euc.append(pos_euc)
        
    # Average across wikitext prompts
    metrics["dynamic"] = {
        "token_embeddings_cosine_similarity_mean": sum(dynamic_wte_cos) / num_prompts,
        "token_embeddings_euclidean_distance_mean": sum(dynamic_wte_euc) / num_prompts,
        "position_embeddings_cosine_similarity_mean": sum(dynamic_wpe_cos) / num_prompts,
        "position_embeddings_euclidean_distance_mean": sum(dynamic_wpe_euc) / num_prompts,
    }
    
    logger.info("[OK] Embedding analysis completed. (Vocabulary similarity: %.4f)", wte_cos)
    return metrics
