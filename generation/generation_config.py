"""
================================================================================
generation/generation_config.py
================================================================================

Project: Anchor-Regularized Model Collapse Prevention
Component: Synthetic Data Generation Pipeline (Generation 1)

Purpose:
    Centralized configuration dataclass for synthetic text generation.
    Automatically adapts settings (like batch size and device) based on the
    execution environment (Local CPU vs Colab GPU).

Authors: Deepak (Research Lead)
Created: 2026-07-09
License: MIT
================================================================================
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any

import torch
from .utils import get_project_root, is_colab_environment


@dataclass
class GenerationConfig:
    """Hyperparameters and paths for Generation 1 synthetic data pipeline."""
    
    # Storage and paths
    project_root: Path = get_project_root()
    frozen_anchor_dir: Path = project_root / "checkpoints" / "anchor_model" / "frozen"
    dataset_source: Path = project_root / "data" / "processed" / "clean_wikitext.txt"
    output_dir: Path = project_root / "data" / "synthetic" / "generation_1"
    
    # Scale and Runtime
    max_documents: int = 10000
    random_seed: int = 42
    
    # Environment-adaptive settings
    is_colab: bool = is_colab_environment()
    device: str = "cuda" if is_colab_environment() and torch.cuda.is_available() else "cpu"
    batch_size: int = 128 if is_colab_environment() else 4  # Small batch for local dev
    
    # Prompt Extraction
    prompt_min_tokens: int = 32
    prompt_max_tokens: int = 64
    
    # Text Generation Decoding Parameters
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    max_new_tokens: int = 256
    
    # Meta
    anchor_version: str = "distilgpt2-anchor-v1"
    
    def to_dict(self) -> Dict[str, Any]:
        """Return parameters as a dictionary for saving with metadata."""
        d = asdict(self)
        # Convert Paths to strings for JSON serialization
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d
    
    def save(self, filepath: Path) -> None:
        """Save configuration to JSON."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
