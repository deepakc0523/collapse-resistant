"""
================================================================================
recursive_generation/__init__.py
================================================================================

Recursive Generation module for the Collapse-Resistant Recursive Language Model
Training Framework.

Generates Generation-2 synthetic data by prompting the trained Generation-1
student model with canonical human anchor prefixes.
"""

from recursive_generation.generation_config import GenerationConfig
from recursive_generation.model_loader import ModelLoader
from recursive_generation.prefix_loader import PrefixLoader, PrefixRecord
from recursive_generation.resume_manager import ResumeManager
from recursive_generation.generator import SyntheticGenerator
from recursive_generation.metadata_writer import MetadataWriter
from recursive_generation.visualization import GenerationVisualizer
from recursive_generation.run_recursive_generation import run_recursive_generation
from recursive_generation.verify_recursive_generation import verify_recursive_generation

__all__ = [
    "GenerationConfig",
    "ModelLoader",
    "PrefixLoader",
    "PrefixRecord",
    "ResumeManager",
    "SyntheticGenerator",
    "MetadataWriter",
    "GenerationVisualizer",
    "run_recursive_generation",
    "verify_recursive_generation",
]
