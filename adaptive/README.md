# Adaptive Threshold Engine (ATE)

The **Adaptive Threshold Engine (ATE)** is the autonomous AI decision-making module in the **Collapse-Resistant Recursive Language Model Training Framework**. Situated directly between the **Synthetic Collapse Risk Score (SCRS)** fusion engine and the downstream **Curriculum Controller**, ATE evaluates representation drift and prediction uncertainty metrics to scientifically synthesize a continuous training policy for Generation-$(N+1)$ recursive training.

---

## 1. Primary Objective & Design Philosophy

ATE acts as an autonomous AI training supervisor. Rather than asking a binary question (*"Has the model collapsed?"*), ATE answers:

> **"What should Generation-$(N+1)$ training look like given the health of the recursive learner?"**

### Core Design Requirements
- **No Hard-Coded "Magic" Thresholds**: ATE avoids simplistic step logic such as `if score > 0.70: stop`.
- **Continuous Mathematical Formulation**: All policy hyperparameters (data blend ratios, learning rates, epochs, sampling temperatures, recursive depth allowance) are derived using smooth logistic sigmoids and convex combinations over the complete SCRS metric profile.
- **Strict Separation of Concerns**: ATE **does not** load models, modify checkpoints, run GPU inference, or generate datasets. It strictly reads `scrs_out/scrs_report.json` and produces machine-actionable policy recommendations.

---

## 2. Pipeline Integration

```
  Representation Probe (probe/)
               │
               ▼
   Ensemble Variance Monitor (ensemble/)
               │
               ▼
 Synthetic Collapse Risk Score (scrs/)
               │
               ▼
 Adaptive Threshold Engine (adaptive/)
               │
               ▼
 Generation-(N+1) Policy Report (adaptive_out/adaptive_policy.json)
               │
               ▼
  Curriculum Controller (Future Module)
```

---

## 3. Mathematical Rationale

### Smooth Risk Sensitivity Factor ($S$)
Given the fused SCRS score $\text{SCRS} \in [0, 1]$, ATE computes a generalized smooth risk sensitivity score $S$:

$$S = \sigma(\text{SCRS}; k, x_0) = \frac{1}{1 + e^{-k \cdot (\text{SCRS} - x_0)}}$$

where $k = 6.0$ (steepness) and $x_0 = 0.50$ (midpoint).

### Continuous Dataset Mix Ratios
To mitigate recursive feedback loop degeneration without abrupt cutoff behavior, synthetic data ratio $r_{\text{synthetic}}$ and canonical anchor data ratio $r_{\text{anchor}}$ are derived continuously:

$$r_{\text{synthetic}} = r_{\text{synthetic}}^{\text{max}} - S \cdot (r_{\text{synthetic}}^{\text{max}} - r_{\text{synthetic}}^{\text{min}})$$

$$r_{\text{anchor}} = 1.0 - r_{\text{synthetic}}$$

This guarantees $r_{\text{synthetic}} + r_{\text{anchor}} = 1.0$ at all times.

### Smooth Hyperparameter Scaling
1. **Recommended Learning Rate ($\eta_{\text{rec}}$)**:
   $$\eta_{\text{rec}} = \eta_{\text{min}} + (\eta_{\text{base}} - \eta_{\text{min}}) \cdot (1 - S)^2$$

2. **Recommended Epochs ($E_{\text{rec}}$)**:
   $$E_{\text{rec}} = \text{round}\Big(E_{\text{min}} + (1 - S) \cdot (E_{\text{max}} - E_{\text{min}})\Big)$$

3. **Sampling Temperature ($T_{\text{rec}}$)**:
   $$T_{\text{rec}} = T_{\text{base}} \cdot \Big(1.0 - 0.5 \cdot (0.6 \cdot R_{\text{rep}} + 0.4 \cdot R_{\text{unc}})\Big)$$

4. **Recursive Utility Cutoff ($U$)**:
   $$U(\text{SCRS}) = 1.0 - \sigma(\text{SCRS}; k=12.0, x_0=0.82)$$
   Training continuation flag $\text{continue\_recursive\_training} = \mathbb{I}(U \ge 0.15)$.

---

## 4. Expected File Structure

```
adaptive/
├── __init__.py
├── adaptive_config.py
├── utils.py
├── scrs_loader.py
├── policy_engine.py
├── recommendation_engine.py
├── adaptive_report.py
├── visualization.py
├── verify_adaptive.py
├── run_adaptive.py
└── README.md
```

---

## 5. Output Specification (`adaptive_out/adaptive_policy.json`)

```json
{
    "metadata": {
        "title": "Adaptive Threshold Engine (ATE) Policy Report",
        "framework_version": "1.0",
        "timestamp": "2026-08-21T20:48:00",
        "scrs_source_report": "D:\\collapse-resistant-training\\scrs_out\\scrs_report.json"
    },
    "training_status": "HIGH_RISK",
    "policy": {
        "synthetic_ratio": 0.3541,
        "anchor_ratio": 0.6459,
        "recommended_epochs": 2,
        "recommended_learning_rate": 8.420000e-06,
        "sampling_temperature": 0.4520,
        "max_generation_depth": 2,
        "continue_recursive_training": true,
        "risk_sensitivity_score": 0.8124
    },
    "scrs_summary": {
        "overall_scrs": 0.7444,
        "representation_risk": 0.7084,
        "uncertainty_risk": 0.7983,
        "upstream_risk_label": "High"
    },
    "primary_risk_driver": "unc_probability_variance (contribution: 0.0665)",
    "recommendations": {
        "status_summary": "The model evaluation yields an overall SCRS of 0.7444...",
        "justifications": [ ... ],
        "curriculum_instructions": {
            "target_generation": "Generation-(N+1)",
            "continue_pipeline": true,
            "dataset_synthesis_spec": {
                "synthetic_ratio": 0.3541,
                "anchor_ratio": 0.6459,
                "sampling_temperature": 0.4520,
                "max_depth": 2
            },
            "training_hyperparameters": {
                "epochs": 2,
                "learning_rate": 8.420000e-06
            },
            "risk_mitigation_mode": "HIGH_RISK"
        },
        "mitigation_actions": [ ... ]
    }
}
```

---

## 6. Generated Visualizations (`adaptive_out/plots/`)

1. **`policy_overview.png`**: Comparative bar plot of baseline vs derived policy hyperparameters.
2. **`metric_influence.png`**: Horizontal bar plot showing individual Probe and Ensemble metric influence.
3. **`training_recommendations.png`**: Continuous synthetic vs anchor mix ratio curves and learning rate decay trajectories.
4. **`recursive_pathway.png`**: Visual flowchart illustrating data flow from Probe/Ensemble through SCRS into Generation-$(N+1)$ policy.
5. **`policy_heatmap.png`**: Matrix heatmap displaying hyperparameter intensity across risk dimensions.

---

## 7. Limitations & Curriculum Controller Integration

### Limitations
- ATE depends strictly on the quality and freshness of `scrs_out/scrs_report.json`.
- ATE does not dynamically monitor real-time GPU training metrics (loss curves); this responsibility is delegated to trainer callbacks during active training execution.

### Future Integration
The generated `adaptive_policy.json` contains the `curriculum_instructions` dictionary specifically designed for direct automated ingestion by the **Curriculum Controller** module to create Generation-2 datasets and launch fine-tuning jobs in Google Colab.
