"""
================================================================================
scrs/__init__.py
================================================================================

Synthetic Collapse Risk Score (SCRS) Package.

Combines representation drift metrics (Probe) and prediction uncertainty metrics
(Ensemble) into a single normalized collapse-risk score [0.0, 1.0].
"""

from scrs.scrs_config import SCRSConfig
from scrs.probe_loader import ProbeLoader, ProbeMetrics
from scrs.ensemble_loader import EnsembleLoader, EnsembleMetrics
from scrs.normalization import Normalizer, NormalizedRepresentationMetrics, NormalizedUncertaintyMetrics
from scrs.weighting_engine import WeightingEngine
from scrs.scrs_engine import SCRSEngine, SCRSResult
from scrs.scrs_report import SCRSReportGenerator
from scrs.visualization import SCRSVisualizer

__all__ = [
    "SCRSConfig",
    "ProbeLoader",
    "ProbeMetrics",
    "EnsembleLoader",
    "EnsembleMetrics",
    "Normalizer",
    "NormalizedRepresentationMetrics",
    "NormalizedUncertaintyMetrics",
    "WeightingEngine",
    "SCRSEngine",
    "SCRSResult",
    "SCRSReportGenerator",
    "SCRSVisualizer",
]
