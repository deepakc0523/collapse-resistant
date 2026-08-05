"""
================================================================================
ensemble/__init__.py
================================================================================

Public API surface for the Ensemble Variance Monitor (EVM).

This module measures prediction uncertainty of the Best Student model using
six calibrated uncertainty metrics. It is completely independent of the Probe
(Representation Drift) subsystem and does not compare against the Anchor model.

It is designed to be a direct upstream input to SCRS — all reported metrics
are normalized to [0, 1] with normalization metadata embedded in output JSON.

Usage
-----
    python -m ensemble.run_ensemble       # Full pipeline
    python -m ensemble.verify_ensemble    # Sanity checks
"""

from .ensemble_config import EnsembleConfig
from .run_ensemble import main as run
from .verify_ensemble import run_verification as verify

__all__ = ["EnsembleConfig", "run", "verify"]
__version__ = "1.0.0"
