"""
================================================================================
probe/cka.py
================================================================================

Linear Centered Kernel Alignment (CKA) computation module for representation validity control.

Measures layer-wise representation similarity between Anchor and Student models
using linear CKA:

    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

where X and Y are mean-centered activation matrices across tokens/samples.
"""

import torch
from typing import Dict, List, Any, Optional

def compute_linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-9) -> float:
    """
    Computes Linear CKA between representation matrices X and Y.
    
    Args:
        x: Activation tensor of shape [N, D1] (N samples/tokens, D1 features).
        y: Activation tensor of shape [N, D2] (N samples/tokens, D2 features).
        eps: Small epsilon to prevent division by zero.
        
    Returns:
        float: Linear CKA score bounded in [0.0, 1.0].
    """
    if x.ndim != 2 or y.ndim != 2:
        x = x.view(-1, x.size(-1))
        y = y.view(-1, y.size(-1))
        
    if x.size(0) != y.size(0):
        min_len = min(x.size(0), y.size(0))
        x = x[:min_len]
        y = y[:min_len]
        
    if x.size(0) <= 1:
        return 1.0

    # Mean-center columns
    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)
    
    # Compute cross-covariance Frobenius norm squared: ||Y^T X||_F^2
    cross_cov = torch.matmul(y_centered.t(), x_centered)
    hsic_xy = torch.sum(cross_cov ** 2)
    
    # Self-covariance Frobenius norm squared
    self_cov_x = torch.matmul(x_centered.t(), x_centered)
    hsic_xx = torch.sum(self_cov_x ** 2)
    
    self_cov_y = torch.matmul(y_centered.t(), y_centered)
    hsic_yy = torch.sum(self_cov_y ** 2)
    
    denom = torch.sqrt(hsic_xx * hsic_yy) + eps
    cka_val = (hsic_xy / denom).item()
    
    return max(0.0, min(1.0, float(cka_val)))


def analyze_layer_wise_cka(
    anchor_features: Dict[str, Any],
    student_features: Dict[str, Any],
    num_layers: int = 6
) -> Dict[str, Any]:
    """
    Computes layer-wise linear CKA similarity between Anchor and Student hidden states.
    
    Args:
        anchor_features: Features extracted by HiddenStateExtractor from Anchor model.
        student_features: Features extracted by HiddenStateExtractor from Student model.
        num_layers: Number of transformer layer blocks (excluding embedding layer 0).
        
    Returns:
        Dict[str, Any]: Layer-wise CKA results and mean CKA.
    """
    anchor_hidden_prompts = anchor_features["hidden_states"]
    student_hidden_prompts = student_features["hidden_states"]
    
    num_prompts = len(anchor_hidden_prompts)
    if num_prompts == 0:
        return {"layer_cka": {}, "mean_cka": 0.0}
        
    # Layer count: typically 0..6 (where 0 is embedding, 1..6 are blocks)
    total_layers = min(
        len(anchor_hidden_prompts[0]),
        len(student_hidden_prompts[0])
    )
    
    layer_cka_scores: Dict[int, List[float]] = {l: [] for l in range(total_layers)}
    
    for prompt_idx in range(num_prompts):
        anchor_prompt_hs = anchor_hidden_prompts[prompt_idx]
        student_prompt_hs = student_hidden_prompts[prompt_idx]
        
        for layer_idx in range(total_layers):
            a_hs = anchor_prompt_hs[layer_idx]
            s_hs = student_prompt_hs[layer_idx]
            
            score = compute_linear_cka(a_hs, s_hs)
            layer_cka_scores[layer_idx].append(score)
            
    layer_wise_mean: Dict[str, float] = {}
    all_scores = []
    
    for layer_idx in range(total_layers):
        scores = layer_cka_scores[layer_idx]
        mean_score = float(sum(scores) / len(scores)) if scores else 0.0
        layer_label = f"layer_{layer_idx}" if layer_idx > 0 else "embedding_layer"
        layer_wise_mean[layer_label] = mean_score
        all_scores.append(mean_score)
        
    overall_mean = float(sum(all_scores) / len(all_scores)) if all_scores else 0.0
    
    return {
        "layer_wise_cka": layer_wise_mean,
        "mean_cka": overall_mean,
        "num_prompts_evaluated": num_prompts,
        "num_layers_evaluated": total_layers
    }
