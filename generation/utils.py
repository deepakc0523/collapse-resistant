"""
================================================================================
generation/utils.py
================================================================================

Project: Anchor-Regularized Model Collapse Prevention
Component: Synthetic Data Generation Pipeline (Generation 1)

Purpose:
    Provides core utilities for the synthetic generation pipeline, including:
    - Environment detection (Local CPU vs. Google Colab GPU)
    - Safe cross-platform path resolution
    - Standardized logging matching the Anchor pipeline

Authors: Deepak (Research Lead)
Created: 2026-07-09
License: MIT
================================================================================
"""

import io
import logging
import os
import sys
from pathlib import Path


def _reconfigure_stdout_utf8() -> None:
    """
    Force UTF-8 encoding on standard output.
    Deep Learning Concept:
        Text generation outputs diverse unicode characters (especially from
        web-scraped corpora like WikiText-103). Windows environments often
        default to cp1252, leading to catastrophic UnicodeEncodeError crashes
        mid-generation. This safely forces UTF-8.
    """
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )

# Execute immediately on import
_reconfigure_stdout_utf8()


def is_colab_environment() -> bool:
    """
    Detect if the code is executing within a Google Colab environment.
    
    Algorithm:
        Checks for the presence of the 'COLAB_GPU' environment variable,
        which is uniquely injected by Google Colab runtimes.
        
    Returns:
        bool: True if running in Colab, False otherwise (e.g., local Windows).
    """
    return "COLAB_GPU" in os.environ


def get_project_root() -> Path:
    """
    Safely resolve the absolute path to the project root directory.
    
    Returns:
        Path: The absolute Path object pointing to the root of the repository.
    """
    # __file__ is generation/utils.py
    # .parent is generation/
    # .parent.parent is the project root
    return Path(__file__).resolve().parent.parent


def setup_logger(name: str, log_file: Path = None) -> logging.Logger:
    """
    Build a standardized professional logger matching the Anchor pipeline.
    
    Args:
        name (str): The name of the module/logger.
        log_file (Path, optional): If provided, also routes logs to this file.
        
    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Avoid attaching handlers multiple times if instantiated repeatedly
    if logger.handlers:
        return logger
        
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    # File Handler (Optional, useful for long GPU runs on Colab)
    if log_file is not None:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)  # Capture deeper logs in file
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        
    return logger
