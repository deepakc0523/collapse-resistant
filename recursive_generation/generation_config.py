"""
================================================================================
recursive_generation/generation_config.py
================================================================================

Configuration dataclass for the Recursive Generation module.

All generation hyperparameters are centralized here. Designed for
Tesla T4 execution inside Google Colab.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Resolve project root relative to this file
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent


@dataclass
class GenerationConfig:
    """
    Configuration for Generation-2 synthetic data production.

    Attributes
    ----------
    student_checkpoint_path : Path
        Path to the trained Generation-1 student model checkpoint.
    prefix_dataset_path : Path
        Path to the same human-cleaned prefix text used in Generation-1.
    output_dir : Path
        Root output directory for all generation artifacts.
    generation_output_dir : Path
        Subdirectory for generation_2 JSONL and metadata.
    plots_dir : Path
        Subdirectory for visualization plots.
    checkpoint_dir : Path
        Directory for intermediate resume checkpoints.
    output_jsonl_path : Path
        Final export path for generation_2_synthetic.jsonl.
    metadata_json_path : Path
        Path to generation_metadata.json.
    summary_txt_path : Path
        Path to generation_summary.txt.
    resume_checkpoint_path : Path
        Path to the intermediate resume checkpoint JSONL (for Colab recovery).
    generation_number : int
        Generation identifier (2 for this module).
    parent_student : str
        Identifier of the parent student model.
    temperature : float
        Sampling temperature for autoregressive generation.
    top_k : int
        Top-K candidates considered per generation step.
    top_p : float
        Top-P (nucleus sampling) probability mass.
    repetition_penalty : float
        Penalty factor for repeated token sequences.
    max_new_tokens : int
        Maximum number of new tokens to generate per prompt.
    min_new_tokens : int
        Minimum number of new tokens to generate per prompt.
    num_beams : int
        Number of beams for beam search (1 = greedy/sampling).
    do_sample : bool
        Whether to use sampling (True) or greedy decoding (False).
    batch_size : int
        Number of prompts processed in a single forward pass.
    max_prefix_tokens : int
        Maximum token length for prefix truncation.
    random_seed : int
        Global random seed for deterministic generation.
    checkpoint_every : int
        Save resume checkpoint after this many generations.
    device : str
        Target compute device ('cuda' for Colab GPU, 'cpu' for local).
    use_amp : bool
        Whether to use Automatic Mixed Precision (AMP) on GPU.
    """

    # --- Paths ---
    student_checkpoint_path: Path = _PROJECT_ROOT / "checkpoints" / "student_model" / "best"
    prefix_dataset_path: Path = _PROJECT_ROOT / "data" / "processed" / "clean_wikitext.txt"
    output_dir: Path = _PROJECT_ROOT / "recursive_generation_out"
    generation_output_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "recursive_generation_out" / "generation_2"
    )
    plots_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "recursive_generation_out" / "plots"
    )
    checkpoint_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "recursive_generation_out" / "resume_checkpoints"
    )
    output_jsonl_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "recursive_generation_out" / "generation_2" / "generation_2_synthetic.jsonl"
    )
    metadata_json_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "recursive_generation_out" / "generation_2" / "generation_metadata.json"
    )
    summary_txt_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "recursive_generation_out" / "generation_2" / "generation_summary.txt"
    )
    resume_checkpoint_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "recursive_generation_out" / "resume_checkpoints" / "resume_state.json"
    )

    # --- Generation Identity ---
    generation_number: int = 2
    parent_student: str = "generation_1"

    # --- Sampling Hyperparameters ---
    temperature: float = 0.70
    top_k: int = 50
    top_p: float = 0.90
    repetition_penalty: float = 1.3
    max_new_tokens: int = 128
    min_new_tokens: int = 20
    num_beams: int = 1
    do_sample: bool = True

    # --- Batching & Prefix ---
    batch_size: int = 8
    max_prefix_tokens: int = 64
    random_seed: int = 42

    # --- Checkpointing ---
    checkpoint_every: int = 500

    # --- Device & Precision ---
    device: str = "cuda"
    use_amp: bool = True

    def __post_init__(self) -> None:
        """Create required directories on initialization."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generation_output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
