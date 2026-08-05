# Ensemble Variance Monitor (EVM)

## Overview

The **Ensemble Variance Monitor (EVM)** is Phase 2 of the Recursive Synthetic Learning Collapse Prevention Framework. It measures the **prediction uncertainty** of the Best Student model using six calibrated, normalized uncertainty metrics.

This module is **completely independent** — it does not compare against the Anchor model and does not calculate collapse risk. Its output is designed to be a direct upstream input to **SCRS (Synthetic Collapse Risk Score)**.

---

## Pipeline Position

```
Representation Drift Analysis (Probe)     ← COMPLETE
            ↓
Ensemble Variance Monitor (EVM)           ← THIS MODULE
            ↓
SCRS (Synthetic Collapse Risk Score)      ← NEXT PHASE
```

---

## Module Structure

```
ensemble/
│
├── __init__.py              # Public API surface
├── ensemble_config.py       # Configuration dataclass (no thresholds)
├── utils.py                 # Logging, device selection, timers
├── model_loader.py          # Student-only model loading (eval, frozen)
├── prompt_loader.py         # Deterministic wikitext prompt sampling
├── probability_extractor.py # Forward pass → softmax probabilities + MC Dropout
├── uncertainty_metrics.py   # Six normalized uncertainty metrics
├── variance_report.py       # SCRS-ready JSON + text report
├── visualization.py         # Five publication-quality figures
├── verify_ensemble.py       # Nine-phase sanity checker
├── run_ensemble.py          # Main entry point
└── README.md                # This file
```

---

## The Six Uncertainty Metrics

All metrics are normalized to **[0, 1]** with normalization derivations documented in code.

| # | Metric | Formula | Range | Interpretation |
|---|--------|---------|-------|----------------|
| 1 | **Predictive Entropy** | `H(p) / log(V)` | [0, 1] | 1 = max uncertain, 0 = max confident |
| 2 | **Top-1 Confidence** | `max(p)` | [0, 1] | Probability of the predicted token |
| 3 | **Top-5 Confidence Spread** | `p₁ - p₅` | [0, 1] | Mass concentration at top-5 |
| 4 | **Probability Variance** | `Var(top-k) / Var_max` | [0, 1] | Distribution sharpness |
| 5 | **Confidence Margin** | `p₁ - p₂` | [0, 1] | Gap between top-1 and top-2 |
| 6 | **MC Dropout Consistency** | Agreement rate across N passes | [0, 1] | Prediction stability under dropout |

### Metric Details

**1. Predictive Entropy**
Shannon entropy measures how spread out the probability mass is. Normalized by `log(vocab_size)` so it is vocabulary-size-agnostic.

**2. Top-1 Confidence**
The maximum probability assigned by the model at each token position, averaged over the prompt sequence. A native probability — no additional normalization needed.

**3. Top-5 Confidence Spread**
Measures how quickly probability mass falls off across the top-5 ranked tokens. A large spread (p₁ ≫ p₅) signals concentration; a small spread signals diffusion.

**4. Probability Variance**
Variance of the top-k probability values, normalized by the theoretical maximum variance `(1/k)(1 - 1/k)`. High variance = peaked = confident; low variance = flat = uncertain.

**5. Confidence Margin**
The difference between the top-1 and top-2 predicted token probabilities. A near-zero margin signals genuine ambiguity between two tokens.

**6. Monte-Carlo Dropout Consistency**
Runs N stochastic forward passes with dropout enabled (via `model.train()`) while keeping `torch.no_grad()` and all parameters frozen. Measures the fraction of passes that agree with the majority-vote prediction. Low consistency → high epistemic uncertainty.

---

## Output Format

```
ensemble_out/
├── variance_report.json       # SCRS-ready structured report
├── variance_summary.txt       # Human-readable summary
└── plots/
    ├── entropy_distribution.png
    ├── confidence_histogram.png
    ├── variance_histogram.png
    ├── confidence_margin.png
    └── mc_dropout_consistency.png
```

### JSON Schema (SCRS Contract)

```json
{
  "metadata": {
    "title": "...",
    "module": "evm",
    "framework_version": "1.0",
    "timestamp": "ISO-8601",
    "configurations": { ... },
    "normalization_notes": {
      "predictive_entropy": "...",
      "top1_confidence": "...",
      ...
    }
  },
  "aggregate_metrics": {
    "mean_predictive_entropy":     0.0,
    "std_predictive_entropy":      0.0,
    "mean_top1_confidence":        0.0,
    "std_top1_confidence":         0.0,
    "mean_top5_confidence_spread": 0.0,
    "std_top5_confidence_spread":  0.0,
    "mean_probability_variance":   0.0,
    "std_probability_variance":    0.0,
    "mean_confidence_margin":      0.0,
    "std_confidence_margin":       0.0,
    "mean_mc_dropout_consistency": 0.0,
    "std_mc_dropout_consistency":  0.0
  },
  "per_prompt_metrics": [
    {
      "prompt_index": 0,
      "predictive_entropy":     0.0,
      "top1_confidence":        0.0,
      "top5_confidence_spread": 0.0,
      "probability_variance":   0.0,
      "confidence_margin":      0.0,
      "mc_dropout_consistency": 0.0
    }
  ]
}
```

---

## Commands

### Verify the module
```bash
python -m ensemble.verify_ensemble
```

Runs a nine-phase sanity check on a single prompt. Does not affect production output.

### Run the full pipeline
```bash
python -m ensemble.run_ensemble
```

Processes up to 100 prompts and writes all artifacts to `ensemble_out/`.

---

## Configuration

Edit `ensemble/ensemble_config.py` to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_prompts` | 100 | Number of prompts to evaluate |
| `batch_size` | 2 | Prompts per forward pass |
| `mc_dropout_passes` | 10 | Number of MC Dropout stochastic passes |
| `top_k_confidence` | 5 | Top-k tokens for spread and variance |
| `device` | `"auto"` | `"auto"`, `"cuda"`, `"cpu"`, or `"mps"` |
| `random_seed` | 42 | Deterministic prompt sampling seed |

---

## Design Constraints

- ❌ Does **NOT** load the Anchor model
- ❌ Does **NOT** calculate collapse risk
- ❌ Does **NOT** apply any thresholds
- ❌ Does **NOT** combine metrics into a score
- ✅ All metrics normalized to [0, 1]
- ✅ `normalization_notes` in JSON so SCRS understands each field's scale
- ✅ Field names are stable for SCRS consumption
- ✅ `per_prompt_metrics` and `aggregate_metrics` both present
- ✅ CPU/GPU compatible, memory-efficient batching
