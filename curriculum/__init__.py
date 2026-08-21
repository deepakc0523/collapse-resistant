"""
================================================================================
curriculum/__init__.py
================================================================================

Curriculum Controller (CC) Module for Recursive Language Model Training Framework.

Converts derived Adaptive Threshold Engine (ATE) policy into progressive,
collapse-resistant Generation-(N+1) training datasets.
"""

from curriculum.curriculum_config import CurriculumConfig
from curriculum.policy_loader import PolicyLoader, ATEPolicyData
from curriculum.dataset_loader import DatasetLoader, DatasetRecord
from curriculum.dataset_sampler import DatasetSampler
from curriculum.dataset_mixer import DatasetMixer, MixedDatasetPool
from curriculum.curriculum_scheduler import CurriculumScheduler, ScheduledCurriculum
from curriculum.metadata_generator import MetadataGenerator
from curriculum.dataset_validator import DatasetValidator, ValidationReport
from curriculum.dataset_exporter import DatasetExporter
from curriculum.curriculum_report import CurriculumReportGenerator
from curriculum.visualization import CurriculumVisualizer
from curriculum.run_curriculum import run_curriculum_pipeline
from curriculum.verify_curriculum import verify_curriculum_module

__all__ = [
    "CurriculumConfig",
    "PolicyLoader",
    "ATEPolicyData",
    "DatasetLoader",
    "DatasetRecord",
    "DatasetSampler",
    "DatasetMixer",
    "MixedDatasetPool",
    "CurriculumScheduler",
    "ScheduledCurriculum",
    "MetadataGenerator",
    "DatasetValidator",
    "ValidationReport",
    "DatasetExporter",
    "CurriculumReportGenerator",
    "CurriculumVisualizer",
    "run_curriculum_pipeline",
    "verify_curriculum_module",
]
