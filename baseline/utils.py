"""
================================================================================
baseline/utils.py
================================================================================

Utility helpers for the Baseline Dataset Builder module.
"""

import hashlib
import logging
import random
from typing import Optional


def get_baseline_logger(name: str = "baseline") -> logging.Logger:
    """Returns configured logger for the baseline module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)


def compute_text_hash(text: str) -> str:
    """Computes short SHA256 hash for text record identification."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
