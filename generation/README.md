# Generation 1 Synthetic Data Pipeline

This directory contains the production-grade pipeline for generating the Generation 1 synthetic dataset using the frozen Anchor Model (DistilGPT2).

## Overview
The goal of this pipeline is to generate exactly 10,000 synthetic documents. These documents represent the "synthetic" dataset produced by the Anchor Model and will later be used to train the Generation 1 Student Model.

## Architecture
The pipeline is designed to be highly modular and environment-agnostic. It seamlessly handles both local debugging (Windows/CPU) and full-scale generation (Google Colab/GPU).

- `generation_config.py`: Centralized configuration.
- `prompt_sampler.py`: Securely extracts real prefixes from WikiText-103.
- `generator.py`: Core batched inference engine.
- `save_dataset.py`: Handles HuggingFace Arrow storage.
- `verify_generation.py`: Data integrity gatekeeper.
- `statistics.py`: Calculates post-generation metrics.
- `utils.py`: Logging and environment detection.

## Execution Environments

### Mode 1: Local Development (Windows / Antigravity IDE)
By default, the pipeline runs in development mode. It will safely execute on CPU with small batches for debugging.

```bash
python generation/run_generation.py
```

### Mode 2: Google Colab (GPU)
To run large-scale generation on Google Colab:
1. Push the latest project codebase to GitHub.
2. Open Google Colab and clone the repository.
3. Install requirements (`pip install torch transformers datasets tqdm`).
4. Mount or upload the trained Frozen Anchor checkpoint to `checkpoints/anchor_model/frozen/`.
5. Run the exact same command:
```bash
python generation/run_generation.py
```
6. Compress the resulting dataset `data/synthetic/generation_1/` into a ZIP and download it back to the local project.

## Provenance and Reproducibility
Every generated document is stored in Apache Arrow format with complete metadata. This ensures that any phenomena observed during later training (such as model collapse) can be fully audited back to the exact prompt, hyperparameter configuration, and anchor model version that produced it.
