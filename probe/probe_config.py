"""
================================================================================
probe/probe_config.py
================================================================================

Configuration settings for the Representation Drift Analysis Framework (PRDAF).
Defines paths, devices, and analysis parameters.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

# Ensure project root is in path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent

@dataclass
class ProbeConfig:
    """Configuration class for the Representation Drift Analysis."""
    
    # Model checkpoints
    anchor_model_path: Path = _PROJECT_ROOT / "checkpoints" / "anchor_model" / "frozen"
    student_model_path: Path = _PROJECT_ROOT / "checkpoints" / "student_model" / "best"
    
    # Dataset prompt source
    dataset_source: Path = _PROJECT_ROOT / "data" / "processed" / "clean_wikitext.txt"
    
    # Output directories
    output_dir: Path = _PROJECT_ROOT / "probe_out"
    plots_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "probe_out" / "plots")
    metrics_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "probe_out" / "metrics")
    logs_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "probe_out" / "logs")
    
    # Report filenames
    report_json_path: Path = field(default_factory=lambda: _PROJECT_ROOT / "probe_out" / "representation_drift_report.json")
    summary_txt_path: Path = field(default_factory=lambda: _PROJECT_ROOT / "probe_out" / "representation_summary.txt")
    
    # Hardware Configuration
    device: str = "auto"  # 'auto', 'cuda', 'cpu', or 'mps'
    
    # Extraction/sampling parameters
    batch_size: int = 2
    max_prompts: int = 10
    prompt_min_tokens: int = 32
    prompt_max_tokens: int = 64
    random_seed: int = 42
    
    # Layers to extract (DistilGPT2 has layers 0 to 5, total 6)
    num_layers: int = 6
    
    def __post_init__(self) -> None:
        """Create necessary directories after initialization."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
