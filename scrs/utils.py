"""
================================================================================
scrs/utils.py
================================================================================

Shared utility functions for the Synthetic Collapse Risk Score (SCRS) module.

Provides:
  - UTF-8 terminal configuration for Windows compatibility
  - Structured logging helper
  - Timing decorator
"""

import io
import sys
import time
import logging
from pathlib import Path
from typing import Callable, Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def setup_utf8_terminal() -> None:
    """Reconfigures stdout/stderr to emit UTF-8 for Windows compatibility."""
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    if hasattr(sys.stderr, "buffer") and getattr(sys.stderr, "encoding", "").lower() != "utf-8":
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )


setup_utf8_terminal()


def get_scrs_logger(name: str = "scrs", log_file: Path = None) -> logging.Logger:
    """
    Creates or retrieves a named logger for SCRS with standardized formatting.

    Args:
        name:     Logger name.
        log_file: Optional path for persistent log file.

    Returns:
        Configured logging.Logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def timed_action(action_name: str, logger: logging.Logger) -> Callable[[F], F]:
    """Decorator factory to log execution time of function."""
    def decorator(func: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            logger.debug("Starting: %s", action_name)
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info("Completed: %s in %.3fs", action_name, elapsed)
            return result
        return wrapper  # type: ignore[return-value]
    return decorator
