"""
================================================================================
ensemble/utils.py
================================================================================

Shared utility functions for the Ensemble Variance Monitor (EVM).

Provides:
  - UTF-8 terminal configuration (required for Windows compatibility)
  - Structured logging with optional file output
  - Hardware device selection (CUDA / MPS / CPU)
  - CUDA memory diagnostics
  - Execution timing decorator
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
    pass  # Handled at call site; torch is a hard dependency

F = TypeVar("F", bound=Callable[..., Any])


def setup_utf8_terminal() -> None:
    """
    Reconfigures stdout/stderr to emit UTF-8 on Windows terminals and notebooks
    where the default encoding may be cp1252 or similar.
    """
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    if hasattr(sys.stderr, "buffer") and getattr(sys.stderr, "encoding", "").lower() != "utf-8":
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )


# Apply immediately when this module is imported
setup_utf8_terminal()


def get_ensemble_logger(name: str, log_file: Path = None) -> logging.Logger:
    """
    Creates or retrieves a named logger with standardised formatting.

    Args:
        name:     Logger name (typically the module dotted path).
        log_file: Optional path for a persistent debug-level log file.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if the logger was already initialised
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # Optional file handler — DEBUG and above
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def select_device(preferred_device: str = "auto", logger: logging.Logger = None) -> "torch.device":
    """
    Resolves the torch compute device from a preference string.

    Args:
        preferred_device: One of 'auto', 'cuda', 'cpu', 'mps'.
        logger:           Logger to emit the selection message.

    Returns:
        torch.device
    """
    if preferred_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        preferred_device == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    elif preferred_device == "cpu":
        device = torch.device("cpu")
    else:
        # 'auto' — pick best available
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if logger is not None:
        if device.type == "cuda":
            logger.info("Selected Compute Device: CUDA (%s)", torch.cuda.get_device_name(0))
        elif device.type == "mps":
            logger.info("Selected Compute Device: Apple MPS")
        else:
            logger.warning(
                "Selected Compute Device: CPU — operations may be slow for large models."
            )

    return device


def get_cuda_memory_report() -> str:
    """
    Returns a formatted string with current CUDA memory usage statistics.
    Returns a 'not available' message if CUDA is not active.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return (
            f"Allocated: {allocated:.2f} MB | "
            f"Reserved:  {reserved:.2f} MB | "
            f"Peak:      {max_allocated:.2f} MB"
        )
    return "CUDA is not active / available."


def timed_action(action_name: str, logger: logging.Logger) -> Callable[[F], F]:
    """
    Decorator factory that logs the execution duration of the wrapped function.

    Args:
        action_name: Human-readable name of the timed action.
        logger:      Logger to emit timing info.

    Returns:
        Decorator that wraps the function with timing instrumentation.
    """
    def decorator(func: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            logger.debug("Starting: %s", action_name)
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info("Completed: %s in %.3f seconds", action_name, elapsed)
            return result
        return wrapper  # type: ignore[return-value]
    return decorator
