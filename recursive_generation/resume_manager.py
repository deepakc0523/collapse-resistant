"""
================================================================================
recursive_generation/resume_manager.py
================================================================================

Manages intermediate checkpointing and resume state for Colab-resilient
Generation-2 synthesis.

Every `checkpoint_every` successful generations, the current progress
is persisted to resume_checkpoints/resume_state.json. On restart,
the engine can skip already-generated prompt indices, avoiding
duplicate regeneration.
"""

import json
import logging
from pathlib import Path
from typing import Set, Optional

from recursive_generation.generation_config import GenerationConfig
from recursive_generation.utils import get_generation_logger


class ResumeManager:
    """
    Tracks which prompt indices have been successfully generated and written.
    Enables seamless resume after Colab disconnections.
    """

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or GenerationConfig()
        self.logger = logger or get_generation_logger("recursive_generation.resume_manager")
        self._completed_indices: Set[int] = set()
        self._total_attempted: int = 0
        self._total_failed: int = 0

    def load_state(self) -> Set[int]:
        """
        Loads previously completed prompt indices from the resume checkpoint.

        Returns
        -------
        Set[int]
            Set of completed prompt indices to skip on resume.
        """
        path = self.config.resume_checkpoint_path
        if not path.exists():
            self.logger.info("No resume checkpoint found. Starting fresh generation.")
            return set()

        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            completed = set(state.get("completed_indices", []))
            self._total_attempted = state.get("total_attempted", 0)
            self._total_failed = state.get("total_failed", 0)
            self.logger.info(
                "Resumed from checkpoint: %d prompts already completed. Skipping...",
                len(completed),
            )
            self._completed_indices = completed
            return completed
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning("Resume checkpoint corrupted (%s). Starting fresh.", str(e))
            return set()

    def mark_completed(self, index: int) -> None:
        """Registers a successfully generated prompt index."""
        self._completed_indices.add(index)
        self._total_attempted += 1

    def mark_failed(self, index: int) -> None:
        """Registers a failed generation attempt."""
        self._total_failed += 1

    def save_state(self) -> None:
        """Persists current completion state to disk."""
        path = self.config.resume_checkpoint_path
        state = {
            "completed_indices": sorted(list(self._completed_indices)),
            "total_attempted": self._total_attempted,
            "total_failed": self._total_failed,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        self.logger.debug("Resume checkpoint saved: %d indices.", len(self._completed_indices))

    @property
    def completed_count(self) -> int:
        return len(self._completed_indices)

    @property
    def failed_count(self) -> int:
        return self._total_failed

    def is_completed(self, index: int) -> bool:
        return index in self._completed_indices
