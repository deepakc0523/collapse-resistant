# Synthetic Collapse Risk Score (SCRS) Framework

## Overview

The **Synthetic Collapse Risk Score (SCRS)** is an independent, non-intrusive mathematical fusion framework. It consumes stored reports from the upstream **Representation Drift Framework (Probe)** and **Ensemble Variance Monitor (EVM)** to compute a unified collapse-risk score bounded between `0.0` (Low Risk / Healthy Model) and `1.0` (High Risk / Severely Collapsed Model).

SCRS functions analogously to a **Credit Score** or **Health Risk Score**. It does not make binary classification decisions or alter training control loops—it quantifies empirical evidence of recursive synthetic collapse across representation drift and predictive uncertainty metrics.

---

## Architecture & Data Flow

```
+------------------------------------+       +------------------------------------+
|  probe_out/                        |       |  ensemble_out/                     |
|  representation_drift_report.json  |       |  variance_report.json              |
+------------------------------------+       +------------------------------------+
                   |                                           |
                   v                                           v
           [ProbeLoader]                               [EnsembleLoader]
                   |                                           |
                   +-------------------+-----------------------+
                                       |
                                       v
                                 [Normalizer]
                          (Risk Scale [0.0, 1.0])
                                       |
                                       v
                              [WeightingEngine]
                     (Group Weights: 60% Rep / 40% Unc)
                                       |
                                       v
                              [SCRSEngine Fusion]
                                       |
                   +-------------------+-----------------------+
                   |                                           |
                   v                                           v
         [SCRSReportGenerator]                         [SCRSVisualizer]
                   |                                           |
                   v                                           v
       scrs_out/scrs_report.json                   scrs_out/plots/
       scrs_out/scrs_summary.txt                   (5 Publication Plots)
```

---

## 1. Normalization Scheme

All raw metrics are converted to a unified **Risk Scale** $r_i \in [0.0, 1.0]$, where $0.0$ represents minimal risk (good alignment / stability) and $1.0$ represents severe risk (collapse / high uncertainty).

### Representation Risk Group (Probe Metrics)
- **Hidden State Drift Risk**: $1.0 - \text{mean\_hidden\_cosine\_similarity}$
- **Embedding Drift Risk**: $1.0 - \text{token\_embeddings\_cosine\_similarity\_mean}$
- **Attention Drift Risk**: $1.0 - \text{mean\_layer\_attention\_cosine\_similarity}$
- **KL Divergence Risk**: Min-Max scaling:
  $$\text{Risk}_{\text{KL}} = \text{clamp}\left(\frac{\text{KL} - \text{KL}_{\min}}{\text{KL}_{\max} - \text{KL}_{\min}}, 0.0, 1.0\right)$$
- **JS Divergence Risk**: Bounded in $[0, 1]$ (scaled by $\ln(2)$ if raw JSD is in nats).
- **Prediction Agreement Risk**: $1.0 - \text{prediction\_agreement\_top1}$

### Uncertainty Risk Group (Ensemble Metrics)
- **Predictive Entropy Risk**: $\text{mean\_predictive\_entropy}$ (naturally normalized to $[0.0, 1.0]$ by $\log(V)$).
- **Top-1 Confidence Risk**: $1.0 - \text{mean\_top1\_confidence}$
- **Top-5 Spread Risk**: $1.0 - \text{mean\_top5\_confidence\_spread}$
- **Probability Variance Risk**: $1.0 - \text{mean\_probability\_variance}$
- **Confidence Margin Risk**: $1.0 - \text{mean\_confidence\_margin}$
- **MC Dropout Consistency Risk**: $1.0 - \text{mean\_mc\_dropout\_consistency}$

---

## 2. Weighting Engine

Group weights and per-metric weights are fully configurable in `scrs_config.py`.

- **Group Weights**:
  - Representation Risk Weight ($W_{\text{rep}}$): `0.60` (60%)
  - Uncertainty Risk Weight ($W_{\text{unc}}$): `0.40` (40%)
- **Metric Weights within Groups**:
  - Equal default distribution ($w_i = \frac{1}{6} \approx 0.1667$) across all 6 metrics within each group.
  - All weight dictionaries are validated to sum strictly to $1.0$.

---

## 3. Mathematical Fusion Formula

1. **Group Risks**:
   $$\text{Representation Risk} = \sum_{i=1}^{6} w_{i, \text{rep}} \cdot r_{i, \text{rep}}$$
   $$\text{Uncertainty Risk} = \sum_{j=1}^{6} w_{j, \text{unc}} \cdot r_{j, \text{unc}}$$

2. **Unified SCRS Score**:
   $$\text{SCRS} = W_{\text{rep}} \times \text{Representation Risk} + W_{\text{unc}} \times \text{Uncertainty Risk}$$

---

## 4. Score Interpretation & Descriptive Labels

SCRS is categorized into 5 continuous risk tiers:

| SCRS Score Range | Risk Category Label | Interpretation |
| :---: | :---: | :--- |
| `0.00 – 0.20` | **Very Low** | Minimal representation drift & strong prediction certainty. |
| `0.20 – 0.40` | **Low** | Slight deviation from anchor; predictions remain stable. |
| `0.40 – 0.60` | **Moderate** | Noticeable representation shift or rising entropy; monitor closely. |
| `0.60 – 0.80` | **High** | Significant collapse risk; high uncertainty and representation degradation. |
| `0.80 – 1.00` | **Critical** | Severe model collapse; representations and logits completely drifted. |

---

## 5. Scope & Limitations

- **Pure Mathematical Fusion**: SCRS does not modify models or issue retraining triggers.
- **Offline Report Processing**: SCRS relies entirely on static, pre-computed `probe_out/` and `ensemble_out/` JSON artifacts.
- **Monotonicity Requirement**: Normalization assumes metric directions are monotonic (e.g. lower similarity always implies higher risk).

---

## Verification & Execution Commands

### Verification Command
Run the standalone verification suite to test loaders, normalization, math bounds, reports, and plots:
```bash
python -m scrs.verify_scrs
```

### Execution Command
Run the main SCRS pipeline:
```bash
python -m scrs.run_scrs
```

Generated outputs will be saved in `scrs_out/`:
- `scrs_out/scrs_report.json`
- `scrs_out/scrs_summary.txt`
- `scrs_out/plots/metric_contribution.png`
- `scrs_out/plots/representation_vs_uncertainty_radar.png`
- `scrs_out/plots/scrs_gauge.png`
- `scrs_out/plots/risk_pie_chart.png`
- `scrs_out/plots/normalized_heatmap.png`
