"""
================================================================================
adaptive/__init__.py
================================================================================

Adaptive Threshold Engine (ATE) Module for Recursive Language Model Training.

Synthesizes scientifically justified Generation-(N+1) training policies from
Synthetic Collapse Risk Score (SCRS) metrics.
"""

from adaptive.adaptive_config import AdaptiveConfig
from adaptive.scrs_loader import SCRSLoader, SCRSData
from adaptive.policy_engine import PolicyEngine, TrainingPolicy, ATEPolicyResult
from adaptive.recommendation_engine import RecommendationEngine, RecommendationReport
from adaptive.adaptive_report import AdaptiveReportGenerator
from adaptive.visualization import AdaptiveVisualizer
from adaptive.run_adaptive import run_adaptive_pipeline
from adaptive.verify_adaptive import verify_adaptive_module

__all__ = [
    "AdaptiveConfig",
    "SCRSLoader",
    "SCRSData",
    "PolicyEngine",
    "TrainingPolicy",
    "ATEPolicyResult",
    "RecommendationEngine",
    "RecommendationReport",
    "AdaptiveReportGenerator",
    "AdaptiveVisualizer",
    "run_adaptive_pipeline",
    "verify_adaptive_module",
]
