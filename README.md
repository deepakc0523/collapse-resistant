# Collapse-Resistant Training (CRT)

A modular framework designed for investigating, monitoring, and preventing representation collapse (e.g., dimensional collapse, complete collapse) in self-supervised learning, contrastive learning, and adversarial training setups.

## Repository Architecture

This codebase is organized into modular components to support clean research and experimentation:

*   **[`anchor/`](file:///d:/collapse-resistant-training/anchor)**: Code and utilities for reference architectures, target models, or anchoring mechanisms to stabilize training representation.
*   **[`trainer/`](file:///d:/collapse-resistant-training/trainer)**: Core training loops, optimizer setups, and training orchestration modules.
*   **[`probe/`](file:///d:/collapse-resistant-training/probe)**: Evaluation probes (e.g., linear probes, cluster analysis, KNN evaluation) to assess representation quality without affecting training.
*   **[`monitor/`](file:///d:/collapse-resistant-training/monitor)**: Metrics collection, dimensional collapse detection, rank estimation, and logging mechanisms.
*   **[`curriculum/`](file:///d:/collapse-resistant-training/curriculum)**: Logic for dynamic curriculum schedules, adaptive loss weighting, or sample selection strategies.
*   **[`api/`](file:///d:/collapse-resistant-training/api)**: Service API layer for running evaluations, querying metrics, or triggering training runs.
*   **[`dashboard/`](file:///d:/collapse-resistant-training/dashboard)**: Real-time visualization dashboard (Streamlit/Gradio-based) for monitoring training stability and representation health.
*   **[`configs/`](file:///d:/collapse-resistant-training/configs)**: YAML/JSON configuration files defining model hyperparameters, datasets, and training options.
*   **[`eval/`](file:///d:/collapse-resistant-training/eval)**: Offline evaluation scripts, downstream task pipelines, and benchmarking suites.
*   **[`notebooks/`](file:///d:/collapse-resistant-training/notebooks)**: Jupyter notebooks for interactive analysis, plotting, and quick experiments.
*   **[`datasets/`](file:///d:/collapse-resistant-training/datasets)**: Data loading, augmentation pipelines, and dataset-specific utilities.

## Getting Started

### Prerequisites

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

## License

MIT
