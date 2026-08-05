"""
================================================================================
ensemble/uncertainty_metrics.py
================================================================================

Six fully implemented uncertainty metrics for the Ensemble Variance Monitor.

All metrics operate on softmax probability tensors produced by the Student
model. No Anchor model is referenced here.

Every metric is normalized to [0, 1] wherever mathematically appropriate.
Normalization constants and their derivations are documented inline.

SCRS downstream contract
------------------------
All values returned by `compute_all_metrics` are scalar floats in [0, 1]
(or documented exceptions). Field names must NOT be changed — SCRS will
consume them by name.

Metrics implemented
-------------------
1. Predictive Entropy          — Overall prediction uncertainty
2. Top-1 Confidence            — Maximum probability assigned to any token
3. Top-5 Confidence Spread     — Probability drop from rank-1 to rank-5
4. Probability Variance        — Sharpness of the top-k distribution
5. Confidence Margin           — Gap between top-1 and top-2 probabilities
6. MC Dropout Consistency      — Stability of predictions under stochastic dropout
"""

import logging
import math
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger("ensemble.uncertainty_metrics")

# Numerical stability floor — prevents log(0) in entropy computations
_EPS: float = 1e-9


# ===========================================================================
# Metric 1 — Predictive Entropy
# ===========================================================================

def compute_predictive_entropy(
    probs: torch.Tensor,
    vocab_size: int,
) -> float:
    """
    Computes the Shannon entropy of a probability distribution and normalizes
    it to [0, 1] by dividing by the theoretical maximum entropy.

    Scientific basis
    ----------------
    For a probability distribution p over V tokens:

        H(p) = -∑_v  p_v · log(p_v)

    Maximum entropy occurs for the uniform distribution:

        H_max = log(V)

    Normalized entropy:

        H_norm(p) = H(p) / log(V)   ∈ [0, 1]

    Interpretation:
      - H_norm → 1: Model is maximally uncertain (uniform distribution).
      - H_norm → 0: Model is maximally confident (all mass on one token).

    Args:
        probs:      Tensor of shape [seq_len, vocab_size] — softmax probabilities.
        vocab_size: Vocabulary size V (used as normalisation denominator).

    Returns:
        Scalar float in [0, 1] representing mean normalised entropy over tokens.
    """
    # Raw entropy per token position: -∑ p_v * log(p_v + eps)
    raw_entropy = -torch.sum(probs * torch.log(probs + _EPS), dim=-1)  # [seq_len]

    # Normalize by log(V) to bound to [0, 1]
    h_max = math.log(vocab_size)
    if h_max < _EPS:
        return 0.0

    normalized = raw_entropy / h_max  # [seq_len]
    return float(normalized.mean().item())


# ===========================================================================
# Metric 2 — Top-1 Confidence
# ===========================================================================

def compute_top1_confidence(probs: torch.Tensor) -> float:
    """
    Computes the mean Top-1 confidence (maximum probability) across tokens.

    Scientific basis
    ----------------
    For each token position t:

        conf_top1(t) = max_v  p_v(t)

    This is the probability assigned to the model's most likely prediction.
    Being a probability, it is already in [0, 1].

    Interpretation:
      - High value → model is very confident in its top prediction.
      - Low value → model spreads probability mass across many tokens.

    Args:
        probs: Tensor of shape [seq_len, vocab_size] — softmax probabilities.

    Returns:
        Scalar float in [0, 1] — mean top-1 confidence over tokens.
    """
    top1_probs = probs.max(dim=-1).values  # [seq_len]
    return float(top1_probs.mean().item())


# ===========================================================================
# Metric 3 — Top-5 Confidence Spread
# ===========================================================================

def compute_top5_confidence_spread(probs: torch.Tensor, top_k: int = 5) -> float:
    """
    Computes the mean spread between the rank-1 and rank-k probabilities.

    Scientific basis
    ----------------
    For each token position t, given the sorted top-k probabilities
    p_(1) ≥ p_(2) ≥ ... ≥ p_(k):

        spread(t) = p_(1)(t) - p_(k)(t)

    A large spread means probability mass is heavily concentrated at the top,
    indicating high confidence. A small spread means the distribution is
    flatter, indicating uncertainty.

    Range: [0, 1] — bounded because both values are probabilities in [0, 1].

    Args:
        probs: Tensor of shape [seq_len, vocab_size] — softmax probabilities.
        top_k: Number of top tokens to consider (default 5).

    Returns:
        Scalar float in [0, 1] — mean top-k spread over tokens.
    """
    k = min(top_k, probs.size(-1))
    top_values, _ = torch.topk(probs, k=k, dim=-1)  # [seq_len, k]

    # Spread: difference between rank-1 and rank-k
    spread = top_values[:, 0] - top_values[:, -1]  # [seq_len]
    return float(spread.mean().item())


# ===========================================================================
# Metric 4 — Probability Variance
# ===========================================================================

def compute_probability_variance(probs: torch.Tensor, top_k: int = 5) -> float:
    """
    Computes the normalized variance of the top-k probability values.

    Scientific basis
    ----------------
    For each token position t, let p_top = [p_(1), ..., p_(k)] be the
    top-k probabilities. The sample variance is:

        Var(p_top) = (1/k) ∑_{i=1}^{k} (p_(i) - mean(p_top))²

    The theoretical maximum variance of a k-element probability vector occurs
    when one element has all the mass and the rest are zero:

        Var_max = (1/k)(1 - 1/k)

    This is derived from: Var = E[p²] - (E[p])²
      E[p] = 1/k, E[p²] = 1/k → Var = 1/k - 1/k² = (1/k)(1 - 1/k)

    Normalized variance:

        Var_norm = Var(p_top) / Var_max   ∈ [0, 1]

    Interpretation:
      - High variance → distribution is peaked (confident).
      - Low variance  → distribution is flat (uncertain).

    Args:
        probs: Tensor of shape [seq_len, vocab_size] — softmax probabilities.
        top_k: Number of top tokens to consider (default 5).

    Returns:
        Scalar float in [0, 1] — mean normalized variance over tokens.
    """
    k = min(top_k, probs.size(-1))
    top_values, _ = torch.topk(probs, k=k, dim=-1)  # [seq_len, k]

    # Sample variance across top-k tokens for each position
    raw_var = top_values.var(dim=-1, unbiased=False)  # [seq_len]

    # Theoretical maximum variance for k items summing to ≤ 1
    var_max = (1.0 / k) * (1.0 - 1.0 / k)

    if var_max < _EPS:
        return 0.0

    normalized = raw_var / var_max
    # Clamp to [0, 1] to guard against floating-point overshoot
    normalized = normalized.clamp(0.0, 1.0)
    return float(normalized.mean().item())


# ===========================================================================
# Metric 5 — Confidence Margin
# ===========================================================================

def compute_confidence_margin(probs: torch.Tensor) -> float:
    """
    Computes the mean margin between the top-1 and top-2 probabilities.

    Scientific basis
    ----------------
    For each token position t:

        margin(t) = p_(1)(t) - p_(2)(t)

    where p_(1) and p_(2) are the two highest probabilities.

    A large margin indicates the model strongly prefers one token over all
    alternatives — a signature of high confidence. A near-zero margin
    indicates the model is confused between two options.

    Range: [0, 1] — both values are probabilities, so their difference
    cannot exceed 1 (achieved only in the degenerate case where p_(1) = 1).

    Args:
        probs: Tensor of shape [seq_len, vocab_size] — softmax probabilities.

    Returns:
        Scalar float in [0, 1] — mean top-1 minus top-2 confidence margin.
    """
    top2_values, _ = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1)  # [seq_len, 2]

    if top2_values.size(-1) < 2:
        # Degenerate vocabulary — no margin possible
        return float(top2_values[:, 0].mean().item())

    margin = top2_values[:, 0] - top2_values[:, 1]  # [seq_len]
    return float(margin.mean().item())


# ===========================================================================
# Metric 6 — Monte-Carlo Dropout Consistency
# ===========================================================================

def compute_mc_dropout_consistency(mc_probs: List[torch.Tensor]) -> float:
    """
    Measures the stability of the model's top-1 token prediction across
    N stochastic forward passes with dropout enabled.

    Scientific basis
    ----------------
    Given N probability distributions [p¹, p², ..., pᴺ] from N MC Dropout
    passes for a single prompt:

    For each token position t:
      1. Compute the top-1 token prediction from each pass:
             y^n(t) = argmax_v  p^n_v(t)

      2. Find the majority-vote prediction:
             y*(t) = mode({y^1(t), ..., y^N(t)})

      3. Compute the agreement rate:
             agreement(t) = (1/N) ∑_n  𝟙[y^n(t) == y*(t)]

    Consistency is the mean agreement rate across all token positions:

        consistency = mean_t  agreement(t)   ∈ [0, 1]

    Interpretation:
      - consistency → 1: All dropout passes agree — model is stable.
      - consistency → 0: Predictions fluctuate across passes — model is
                         highly sensitive to dropout stochasticity (uncertain).

    Args:
        mc_probs: List of N probability tensors, each of shape
                  [seq_len, vocab_size], representing N dropout passes
                  for a single prompt.

    Returns:
        Scalar float in [0, 1] — mean prediction consistency.
    """
    if not mc_probs:
        return 0.0

    n_passes = len(mc_probs)
    seq_len = mc_probs[0].size(0)

    # Collect top-1 token per pass per position
    # predictions shape: [n_passes, seq_len]
    predictions = torch.stack(
        [probs.argmax(dim=-1) for probs in mc_probs], dim=0
    )  # [n_passes, seq_len]

    # Majority vote per position
    agreement_rates: List[float] = []
    for t in range(seq_len):
        token_preds = predictions[:, t].tolist()  # length n_passes

        # Find majority vote
        counts: Dict[int, int] = {}
        for tok in token_preds:
            counts[tok] = counts.get(tok, 0) + 1
        majority_token = max(counts, key=counts.__getitem__)

        # Agreement rate with majority
        agree_count = counts[majority_token]
        agreement_rates.append(agree_count / n_passes)

    return float(np.mean(agreement_rates))


# ===========================================================================
# Aggregated entry point
# ===========================================================================

def compute_all_metrics(
    softmax_probs: List[torch.Tensor],
    mc_probs: List[List[torch.Tensor]],
    top_k: int = 5,
    vocab_size: int = 50257,
) -> Dict[str, Any]:
    """
    Computes all six uncertainty metrics for every prompt and returns both
    per-prompt results and dataset-level aggregates.

    All scalar values are normalized to [0, 1] where applicable.

    Args:
        softmax_probs: List of Tensors [seq_len, vocab_size], one per prompt.
                       From standard (deterministic) forward passes.
        mc_probs:      List of lists. mc_probs[i] is a list of N tensors
                       [seq_len, vocab_size] from N MC Dropout passes for
                       prompt i.
        top_k:         Number of top tokens used in spread/variance metrics.
        vocab_size:    Vocabulary size for entropy normalisation.

    Returns:
        Dict with structure:
          {
            "per_prompt": [
                {
                  "prompt_index": int,
                  "predictive_entropy":    float,  # [0, 1]
                  "top1_confidence":       float,  # [0, 1]
                  "top5_confidence_spread": float, # [0, 1]
                  "probability_variance":  float,  # [0, 1]
                  "confidence_margin":     float,  # [0, 1]
                  "mc_dropout_consistency": float, # [0, 1]
                },
                ...
            ],
            "aggregate": {
                "mean_predictive_entropy":     float,
                "std_predictive_entropy":      float,
                "mean_top1_confidence":        float,
                "std_top1_confidence":         float,
                "mean_top5_confidence_spread": float,
                "std_top5_confidence_spread":  float,
                "mean_probability_variance":   float,
                "std_probability_variance":    float,
                "mean_confidence_margin":      float,
                "std_confidence_margin":       float,
                "mean_mc_dropout_consistency": float,
                "std_mc_dropout_consistency":  float,
            }
          }
    """
    num_prompts = len(softmax_probs)
    if num_prompts == 0:
        raise ValueError("softmax_probs is empty — no prompts to process.")

    logger.info(
        "Computing uncertainty metrics for %d prompts (top_k=%d, vocab_size=%d)...",
        num_prompts,
        top_k,
        vocab_size,
    )

    per_prompt_results: List[Dict[str, Any]] = []

    for idx in range(num_prompts):
        probs = softmax_probs[idx]  # [seq_len, vocab_size]

        # Metric 1: Predictive Entropy
        entropy = compute_predictive_entropy(probs, vocab_size)

        # Metric 2: Top-1 Confidence
        top1_conf = compute_top1_confidence(probs)

        # Metric 3: Top-5 Confidence Spread
        top5_spread = compute_top5_confidence_spread(probs, top_k=top_k)

        # Metric 4: Probability Variance
        prob_var = compute_probability_variance(probs, top_k=top_k)

        # Metric 5: Confidence Margin
        margin = compute_confidence_margin(probs)

        # Metric 6: MC Dropout Consistency
        if mc_probs and idx < len(mc_probs) and mc_probs[idx]:
            mc_consistency = compute_mc_dropout_consistency(mc_probs[idx])
        else:
            logger.warning(
                "No MC Dropout data for prompt %d — consistency set to NaN.", idx
            )
            mc_consistency = float("nan")

        per_prompt_results.append(
            {
                "prompt_index":           idx,
                "predictive_entropy":     entropy,
                "top1_confidence":        top1_conf,
                "top5_confidence_spread": top5_spread,
                "probability_variance":   prob_var,
                "confidence_margin":      margin,
                "mc_dropout_consistency": mc_consistency,
            }
        )

        if (idx + 1) % 10 == 0 or (idx + 1) == num_prompts:
            logger.debug("Computed metrics for %d / %d prompts.", idx + 1, num_prompts)

    # -----------------------------------------------------------------------
    # Aggregate statistics (mean + std across prompts)
    # -----------------------------------------------------------------------
    fields = [
        "predictive_entropy",
        "top1_confidence",
        "top5_confidence_spread",
        "probability_variance",
        "confidence_margin",
        "mc_dropout_consistency",
    ]

    aggregate: Dict[str, float] = {}
    for field in fields:
        values = [
            r[field] for r in per_prompt_results
            if not (isinstance(r[field], float) and math.isnan(r[field]))
        ]
        if values:
            arr = np.array(values, dtype=np.float64)
            aggregate[f"mean_{field}"] = float(arr.mean())
            aggregate[f"std_{field}"] = float(arr.std())
        else:
            aggregate[f"mean_{field}"] = float("nan")
            aggregate[f"std_{field}"] = float("nan")

    logger.info("[OK] All uncertainty metrics computed successfully.")
    return {"per_prompt": per_prompt_results, "aggregate": aggregate}
