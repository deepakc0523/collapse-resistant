"""
================================================================================
probe/utils.py
================================================================================

Utility functions for the Representation Drift Analysis Framework.
Provides robust UTF-8 terminal handling, flexible logging, GPU diagnostics, 
and benchmarking timers.
"""

import io
import sys
import time
import logging
from pathlib import Path
from typing import Callable, Any, TypeVar

try:
    import torch
except ImportError:
    pass

F = TypeVar('F', bound=Callable[..., Any])

def setup_utf8_terminal() -> None:
    """Configures stdout/stderr to support UTF-8 formatting on Windows and notebooks."""
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    if hasattr(sys.stderr, "buffer") and getattr(sys.stderr, "encoding", "").lower() != "utf-8":
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )

# Initialize terminal configuration
setup_utf8_terminal()

def get_probe_logger(name: str, log_file: Path = None) -> logging.Logger:
    """
    Creates or retrieves a logger with standardized console and file output formatting.
    
    Args:
        name: Name of the logger.
        log_file: Optional path to a file where logs should be appended.
        
    Returns:
        A logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Check if handlers already exist to avoid duplication
    if logger.handlers:
        return logger

    # standard output format
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler (logs to stdout at INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # Optional File Handler (logs to file at DEBUG level)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def select_device(preferred_device: str = "auto", logger: logging.Logger = None) -> "torch.device":
    """
    Determines and returns the torch device to use.
    
    Args:
        preferred_device: 'auto', 'cuda', 'cpu', or 'mps'.
        logger: Logger to output the selection info.
        
    Returns:
        torch.device
    """
    if preferred_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif preferred_device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif preferred_device == "cpu":
        device = torch.device("cpu")
    else:
        # 'auto' or preferred device is unavailable
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
    if logger:
        if device.type == "cuda":
            logger.info("Selected Compute Device: CUDA (%s)", torch.cuda.get_device_name(0))
        elif device.type == "mps":
            logger.info("Selected Compute Device: Apple MPS")
        else:
            logger.warning("Selected Compute Device: CPU — operations may be slow.")
            
    return device


def get_cuda_memory_report() -> str:
    """Returns a formatted string representing current CUDA memory utilisation."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return (
            f"Allocated: {allocated:.2f} MB | "
            f"Reserved: {reserved:.2f} MB | "
            f"Max Peak Allocated: {max_allocated:.2f} MB"
        )
    return "CUDA is not active/available."


def timed_action(action_name: str, logger: logging.Logger) -> Callable[[F], F]:
    """Decorator to measure and log the execution time of a function."""
    def decorator(func: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            logger.debug("Starting: %s", action_name)
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info("Completed: %s in %.3f seconds", action_name, duration)
            return result
        return wrapper  # type: ignore
    return decorator
