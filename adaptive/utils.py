"""
================================================================================
adaptive/utils.py
================================================================================

Utility functions, numerical mathematical helper methods, and logging configuration
for the Adaptive Threshold Engine (ATE).
"""

import sys
import math
import logging
from typing import Any, Dict


def get_adaptive_logger(name: str = "adaptive") -> logging.Logger:
    """
    Creates or retrieves a standardized logger for the Adaptive Threshold Engine.

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


def sigmoid(x: float, k: float = 6.0, x0: float = 0.50) -> float:
    """
    Computes generalized logistic sigmoid transfer function:
        S(x) = 1 / (1 + exp(-k * (x - x0)))

    Parameters
    ----------
    x : float
        Input value (e.g. risk score in [0, 1]).
    k : float
        Steepness parameter.
    x0 : float
        Midpoint shift parameter.

    Returns
    -------
    float
        Smoothed output value in (0, 1).
    """
    # Numerically stable logistic function
    z = -k * (x - x0)
    if z > 40.0:
        return 0.0
    elif z < -40.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


def clamp(val: float, min_val: float, max_val: float) -> float:
    """
    Clamps value within specified bounding interval [min_val, max_val].

    Parameters
    ----------
    val : float
        Input value.
    min_val : float
        Lower bound.
    max_val : float
        Upper bound.

    Returns
    -------
    float
        Clamped value.
    """
    return max(min_val, min(max_val, val))
