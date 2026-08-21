"""
================================================================================
curriculum/utils.py
================================================================================

Utility functions, logging configuration, hashing helpers, and random seed control
for the Curriculum Controller (CC).
"""

import sys
import hashlib
import random
import logging
import numpy as np
from typing import Any, Dict


def get_curriculum_logger(name: str = "curriculum") -> logging.Logger:
    """
    Creates or retrieves a standardized logger for the Curriculum Controller.

    Parameters
    ----------
    name : str
        Logger hierarchy name.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def set_seed(seed: int = 42) -> None:
    """
    Sets global Python and NumPy random seeds for deterministic reproducibility.

    Parameters
    ----------
    seed : int
        Random seed integer.
    """
    random.seed(seed)
    np.random.seed(seed)


def compute_text_hash(text: str) -> str:
    """
    Computes MD5 hash snippet of text string for duplicate record detection.

    Parameters
    ----------
    text : str
        Input string content.

    Returns
    -------
    str
        First 16 hex characters of MD5 hash.
    """
    cleaned = text.strip().lower()
    return hashlib.md5(cleaned.encode("utf-8")).hexdigest()[:16]
