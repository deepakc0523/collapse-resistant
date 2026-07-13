"""
================================================================================
probe/similarity_metrics.py
================================================================================

Core mathematical formulations of representation drift, embedding, 
and logit distribution comparison metrics. Uses PyTorch for batched computation.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any

def compute_cosine_similarity(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Computes cosine similarity between two tensors along a given dimension.
    
    Args:
        a, b: Input tensors of matching shapes.
        dim: Dimension along which to compute similarity.
        
    Returns:
        Tensor of similarities (values between -1 and 1).
    """
    return F.cosine_similarity(a, b, dim=dim)


def compute_euclidean_distance(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Computes pairwise Euclidean distance between two tensors.
    
    Args:
        a, b: Input tensors of matching shapes.
        dim: Dimension along which to compute distance.
        
    Returns:
        Tensor of distance values (non-negative).
    """
    return torch.norm(a - b, p=2, dim=dim)


def compute_kl_divergence(
    p_logits: torch.Tensor, 
    q_logits: torch.Tensor, 
    eps: float = 1e-9
) -> torch.Tensor:
    """
    Computes Kullback-Leibler (KL) Divergence from Q to P: KL(P || Q).
    Typically P is the reference Anchor distribution, and Q is the Student.
    
    Args:
        p_logits: Logits representing target distribution P.
        q_logits: Logits representing approximate distribution Q.
        eps: Small constant to avoid log(0) issues.
        
    Returns:
        Tensor of KL divergences.
    """
    # Convert logits to probability distributions
    p_probs = F.softmax(p_logits, dim=-1)
    q_probs = F.softmax(q_logits, dim=-1)
    
    # KL = sum( P * log(P / Q) )
    kl = p_probs * (torch.log(p_probs + eps) - torch.log(q_probs + eps))
    return torch.sum(kl, dim=-1)


def compute_js_divergence(
    p_logits: torch.Tensor, 
    q_logits: torch.Tensor, 
    eps: float = 1e-9
) -> torch.Tensor:
    """
    Computes Jensen-Shannon Divergence (JSD) between P and Q.
    JSD is symmetric and bounded between 0 and 1 (when using base 2 log, 
    but here we use natural log, so bounded between 0 and ln(2)).
    
    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)
    
    Args:
        p_logits: Logits representing distribution P.
        q_logits: Logits representing distribution Q.
        eps: Small constant to avoid log(0) issues.
        
    Returns:
        Tensor of JSD values.
    """
    p_probs = F.softmax(p_logits, dim=-1)
    q_probs = F.softmax(q_logits, dim=-1)
    
    m_probs = 0.5 * (p_probs + q_probs)
    
    # Compute KL(P || M) and KL(Q || M)
    kl_pm = p_probs * (torch.log(p_probs + eps) - torch.log(m_probs + eps))
    kl_qm = q_probs * (torch.log(q_probs + eps) - torch.log(m_probs + eps))
    
    jsd = 0.5 * torch.sum(kl_pm, dim=-1) + 0.5 * torch.sum(kl_qm, dim=-1)
    return jsd


def compute_prediction_agreement(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    top_k: int = 1
) -> torch.Tensor:
    """
    Measures prediction agreement between the two models.
    For top_k = 1, checks if argmax(P) == argmax(Q).
    For top_k > 1, checks if the argmax(Q) is present within top_k(P) or vice versa.
    Let's implement: proportion of top-1 predictions of model Q that match top-1 of P,
    or the intersection rate of their top-k prediction sets.
    
    Args:
        p_logits: Logits of model P.
        q_logits: Logits of model Q.
        top_k: Top-k tokens to consider.
        
    Returns:
        Tensor of agreement flags (0.0 or 1.0 for top_k=1, or fraction of overlap).
    """
    if top_k == 1:
        p_pred = torch.argmax(p_logits, dim=-1)
        q_pred = torch.argmax(q_logits, dim=-1)
        return (p_pred == q_pred).float()
    else:
        # Compute set overlap for top-k elements
        _, p_top = torch.topk(p_logits, k=top_k, dim=-1)
        _, q_top = torch.topk(q_logits, k=top_k, dim=-1)
        
        # We want to know for each position, what fraction of q's top-k tokens are in p's top-k tokens
        # Loop over batch/sequence positions
        # Standard shape of logits: [Batch, SeqLen, Vocab] or flattened [N, Vocab]
        flat_p = p_top.view(-1, top_k)
        flat_q = q_top.view(-1, top_k)
        
        overlaps = []
        for i in range(flat_p.size(0)):
            p_set = set(flat_p[i].tolist())
            q_set = set(flat_q[i].tolist())
            overlap_count = len(p_set.intersection(q_set))
            overlaps.append(overlap_count / top_k)
            
        return torch.tensor(overlaps, device=p_logits.device).view(p_logits.shape[:-1])


def compute_vocabulary_overlap(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    top_k: int = 50
) -> torch.Tensor:
    """
    Calculates the Jaccard Index of the top-k vocabulary choices.
    Jaccard(P_k, Q_k) = |P_k ∩ Q_k| / |P_k ∪ Q_k|
    
    Args:
        p_logits: Logits of model P.
        q_logits: Logits of model Q.
        top_k: Size of token pool.
        
    Returns:
        Tensor representing vocabulary overlap (values between 0.0 and 1.0).
    """
    _, p_top = torch.topk(p_logits, k=top_k, dim=-1)
    _, q_top = torch.topk(q_logits, k=top_k, dim=-1)
    
    flat_p = p_top.view(-1, top_k)
    flat_q = q_top.view(-1, top_k)
    
    overlaps = []
    for i in range(flat_p.size(0)):
        p_set = set(flat_p[i].tolist())
        q_set = set(flat_q[i].tolist())
        intersection = len(p_set.intersection(q_set))
        union = len(p_set.union(q_set))
        overlaps.append(intersection / union if union > 0 else 0.0)
        
    return torch.tensor(overlaps, device=p_logits.device).view(p_logits.shape[:-1])
