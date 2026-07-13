"""
================================================================================
student/utils.py
================================================================================

Utility functions for the Student Model Training Pipeline.
Includes logging configuration and device/environment detection.
"""

import io
import logging
import sys
import random
from pathlib import Path

try:
    import torch
    import numpy as np
except ImportError:
    pass

def _reconfigure_stdout_utf8() -> None:
    """Force UTF-8 on Windows terminals for compatibility."""
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )

_reconfigure_stdout_utf8()

def get_logger(name: str, log_file: Path = None) -> logging.Logger:
    """
    Build a logger that writes to stdout and optionally to a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Avoid adding multiple handlers if called multiple times
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

def get_device(logger: logging.Logger) -> torch.device:
    """
    Select the best available compute device.
    Automatically detects CUDA (Colab/Local GPU), MPS (Mac), or CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Device: CUDA (%s)", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Device: Apple MPS")
    else:
        device = torch.device("cpu")
        logger.warning("Device: CPU — training will be slow.")
    return device

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across runs.
    """
    random.seed(seed)
    if 'numpy' in sys.modules:
        np.random.seed(seed)
    if 'torch' in sys.modules:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
