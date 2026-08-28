# Experimental Validation Suite: Collapse-Resistant Training

This package provides a reproducible, statistical experimental-validation layer for the research paper on model-internal collapse-risk monitoring and adaptive synthetic-data curriculum control.

---

## Overview & Scientific Rationale

### 1. Multi-Seed Validation (`experiments.multi_seed`)
- **Motivation**: Single-seed evaluation may reflect random prompt sampling variations. Multi-seed validation evaluates Student-2 Baseline vs. Student-2 Adaptive across multiple random seeds (default: `42`, `123`, `456`, `789`, `2026`).
- **Statistical Integrity**: Calculates descriptive statistics (mean, std, min, max) and paired statistical tests (paired $t$-test and Wilcoxon signed-rank test). Reports actual $p$-values without fabricating statistical significance.

### 2. Monitoring Component Ablation Study (`experiments.ablation`)
- **Motivation**: Quantifies the individual contribution of PRDAF (representation drift) vs EVM (uncertainty variance) to the unified Synthetic Collapse Risk Score (SCRS).
- **Configurations**:
  - `FULL`: PRDAF (60%) + EVM (40%) combined monitoring.
  - `NO_PRDAF`: EVM uncertainty monitoring only (100% EVM, 0% PRDAF).
  - `NO_EVM`: PRDAF representation monitoring only (100% PRDAF, 0% EVM).

### 3. SCRS Weighting Sensitivity Analysis (`experiments.weighting_sensitivity`)
- **Motivation**: Demonstrates that the framework's conclusions are robust across weighting configurations and not sensitive to a single arbitrary weight pair.
- **Evaluated Weighting Ratios**: `50:50`, `60:40` (production default), `70:30`, and grid across $0.1$ to $0.9$. Evaluated on Student-1, Student-2 Baseline, and Student-2 Adaptive models.

### 4. Linear CKA Representation Validity Control (`experiments.cka`)
- **Motivation**: Provides an independent, modern representation-similarity diagnostic using Linear Centered Kernel Alignment (CKA):
  $$\text{CKA}(X, Y) = \frac{\|Y^T X\|_F^2}{\|X^T X\|_F \|Y^T Y\|_F}$$
- **Isolation**: Implemented as an isolated diagnostic module (`probe.cka`) without replacing existing cosine, KL, JS, or MMD metrics.

---

## Directory Structure

```text
research_results/
    final_validation/
        multi_seed/
            seed_42/
            seed_123/
            seed_456/
            seed_789/
            seed_2026/
            aggregate/
                multi_seed_report.json
                multi_seed_summary.txt
                multi_seed_comparison.csv
                multiseed_scrs_summary.png
                per_seed_scrs_comparison.png
        ablation/
            full/
            no_prdaf/
            no_evm/
            ablation_summary.json
            ablation_summary.csv
            ablation_summary.txt
            ablation_comparison.png
        weighting_sensitivity/
            weighting_sensitivity_report.json
            weighting_sensitivity_summary.csv
            weighting_sensitivity_summary.txt
            scrs_weighting_sensitivity.png
        cka/
            cka_report.json
            cka_summary.csv
            cka_summary.txt
            layer_wise_cka.png
```

Existing paths (`research_results/student1`, `student2_baseline`, `student2_adaptive`) are preserved.

---

## Execution Commands

### Local & Google Colab CLI Usage

All experiment scripts accept configurable CLI arguments via `argparse`.

#### 1. Multi-Seed Validation
```bash
python -m experiments.multi_seed \
    --anchor-model-path checkpoints/anchor_model/frozen \
    --baseline-model-path checkpoints/student_model/baseline \
    --adaptive-model-path checkpoints/student_model/adaptive \
    --dataset-path data/processed/clean_wikitext.txt \
    --output-dir research_results/final_validation/multi_seed \
    --seeds 42 123 456 789 2026
```

#### 2. Component Ablation Study
```bash
python -m experiments.ablation \
    --output-dir research_results/final_validation/ablation \
    --probe-report-path probe_out/representation_drift_report.json \
    --ensemble-report-path ensemble_out/variance_report.json
```

#### 3. Weighting Sensitivity Analysis
```bash
python -m experiments.weighting_sensitivity \
    --output-dir research_results/final_validation/weighting_sensitivity \
    --weight-configs 50:50 60:40 70:30
```

#### 4. Representation Validity CKA
```bash
python -m experiments.cka \
    --anchor-model-path checkpoints/anchor_model/frozen \
    --student-model-path checkpoints/student_model/best \
    --dataset-path data/processed/clean_wikitext.txt \
    --output-dir research_results/final_validation/cka
```

#### 5. Verification Suite
```bash
python -m experiments.verify_experiments
```

---

## Status Classification

- **IMPLEMENTED**: All codebase components, CLI modules (`experiments.multi_seed`, `experiments.ablation`, `experiments.weighting_sensitivity`, `experiments.cka`), CKA diagnostic module (`probe.cka`), visualization generators, and verification test suites are fully implemented and verified locally.
- **EXECUTED**: Full final GPU inference over Google Drive checkpoints will be executed upon Colab notebook launch.
