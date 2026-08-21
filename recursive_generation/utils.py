"""
================================================================================
recursive_generation/utils.py
================================================================================

Utility functions for the Recursive Generation module.

Provides logging configuration, seed control, device resolution,
and safe JSON helpers.
"""

import sys
import json
import random
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import numpy as np


def get_generation_logger(name: str = "recursive_generation") -> logging.Logger:
    """
    Creates or retrieves a standardized logger.

    Parameters
    ----------
    name : str
        Logger namespace.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def set_seed(seed: int = 42) -> None:
    """Sets Python, NumPy, and PyTorch random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preferred: str = "cuda") -> torch.device:
    """
    Resolves compute device, falling back gracefully to CPU.

    Parameters
    ----------
    preferred : str
        Preferred device string ('cuda' or 'cpu').

    Returns
    -------
    torch.device
        Resolved device.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def safe_json_load(path: Path) -> Dict[str, Any]:
    """Safely loads a JSON file, returning an empty dict on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
