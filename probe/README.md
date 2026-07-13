# Representation Drift Analysis Framework (PRDAF)

This module implements the complete, scientific analysis pipeline for evaluating **Representation Drift** and potential **Model Representation Collapse** in the self-training curriculum. It compares the internal representational manifolds and token output policies of the **Frozen Anchor Model** against the newly trained **Student Model** (which was initialized from random weights and trained on Generation 1 synthetic data).

This is a **measurement-only** framework. It does not train models or update weights. Both models run in evaluation mode (`model.eval()`) with gradient computation disabled (`torch.no_grad()`).

---

## 1. Architectural Layout

```
probe/
├── README.md                 # Framework documentation and mathematical formulas
├── probe_config.py           # Configuration parameters and paths
├── utils.py                  # Logger and environment/device utilities
├── model_loader.py           # Evaluation loader for Anchor and Student models
├── prompt_loader.py          # Loader/sampler for wikitext prompts
├── hidden_state_extractor.py # Forward-pass layer activation extractor
├── similarity_metrics.py     # Pure mathematical divergence/similarity formulas
├── embedding_analysis.py     # Token (WTE) and Position (WPE) embedding analyzer
├── attention_analysis.py     # Head-wise and layer-wise attention divergence comparison
├── logit_analysis.py         # Output token policy divergence analyzer
├── drift_report.py           # Scientific JSON and TXT report compilation
├── visualization.py          # Publication-ready diagnostic charts generator
├── verify_probe.py           # Sanity verification checklist script
└── run_probe.py              # Main pipeline execution orchestrator
```

---

## 2. Mathematical Formulations & Metrics

The framework implements and evaluates the following scientific metrics:

### A. Cosine Similarity (Representational Alignment)
Computes the angular alignment between hidden representation vectors $A$ and $B$:
$$\text{CosineSimilarity}(A, B) = \frac{A \cdot B}{\|A\|_2 \|B\|_2}$$
Ranges from $-1.0$ (perfect opposition) to $1.0$ (perfect alignment). Evaluated on static embedding matrices, dynamic contextual activations, and layer-by-layer hidden states.

### B. Euclidean Distance (Geometric Displacement)
Measures the straight-line distance between representations $A$ and $B$:
$$\text{EuclideanDistance}(A, B) = \|A - B\|_2 = \sqrt{\sum_{i=1}^d (A_i - B_i)^2}$$

### C. Kullback-Leibler (KL) Divergence (Information Loss)
Measures the informational divergence from the Student model probability distribution $Q$ to the reference Anchor model distribution $P$ over the vocabulary space:
$$D_{\text{KL}}(P \parallel Q) = \sum_{w \in \mathcal{V}} P(w) \log \left( \frac{P(w)}{Q(w) + \epsilon} \right)$$
Tracks the loss of entropy/contextual policy alignment between models.

### D. Jensen-Shannon Divergence (Symmetric Discrepancy)
A symmetric, bounded version of KL divergence. It uses an average distribution $M = \frac{1}{2}(P + Q)$:
$$D_{\text{JS}}(P \parallel Q) = \frac{1}{2} D_{\text{KL}}(P \parallel M) + \frac{1}{2} D_{\text{KL}}(Q \parallel M)$$
Bounded in $[0, \ln(2)]$ when using natural logarithms.

### E. Prediction Agreement
Tracks the percentage of tokens where the primary output predictions align:
$$\text{Agreement}_{\text{Top-}k} = \frac{1}{N}\sum_{i=1}^N \mathbb{I}(\text{Prediction}_Q(x_i) \cap \text{Top-}k(\text{Prediction}_P(x_i)) \neq \emptyset)$$
Evaluated for Top-1 (identical argmax tokens) and Top-5.

### F. Vocabulary Overlap (Jaccard Index)
Measures similarity between top-$k$ vocabulary pools chosen by both models at each position:
$$J(P_k, Q_k) = \frac{|P_k \cap Q_k|}{|P_k \cup Q_k|}$$
Computed for the Top-50 vocabulary tokens at each sequence position.

---

## 3. Execution Instructions

### Verification Dry-Run
To run a fast sanity check on a single prompt string to verify model loading, feature extraction, and report generation, execute:
```bash
python -m probe.verify_probe
```

### Full Analysis Pipeline
To run the full pipeline over the sampled `clean_wikitext.txt` prompt set, compile output files, and generate diagrams, execute:
```bash
python -m probe.run_probe
```

---

## 4. Framework Outputs

Upon running the analysis, all outputs are written to the `probe_out/` directory:
- **`representation_drift_report.json`**: Complete structured JSON containing static and dynamic embedding metrics, layer hidden states similarities, attention divergences, and logit overlaps.
- **`representation_summary.txt`**: A clean, publication-ready scientific text summary detailing performance comparisons and deep learning remarks.
- **`plots/`**:
  - `representation_drift.png`: Line chart tracking Cosine Similarity and Euclidean Distance across all layers.
  - `attention_head_jsd.png`: Head-wise and layer-wise attention distribution divergence heatmap.
  - `embedding_similarity.png`: Bar chart contrasting parameter weights similarity against active prompt embeddings.
  - `prediction_agreement.png`: Bar chart of Top-1 / Top-5 agreement and top-50 Jaccard overlap.
  - `kl_js_divergence.png`: Comparison of KL vs JS Divergence on logits.
