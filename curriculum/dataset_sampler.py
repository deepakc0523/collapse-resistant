"""
================================================================================
curriculum/dataset_sampler.py
================================================================================

Deterministic dataset sampler module for the Curriculum Controller.

Ensures seed-reproducible random sampling and stratified sampling across anchor
and synthetic dataset records.
"""

import random
import logging
from typing import List, Optional

from curriculum.curriculum_config import CurriculumConfig
from curriculum.dataset_loader import DatasetRecord
from curriculum.utils import get_curriculum_logger, set_seed


class DatasetSampler:
    """Provides seed-reproducible sampling operations over dataset collections."""

    def __init__(
        self,
        config: Optional[CurriculumConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or CurriculumConfig()
        self.logger = logger or get_curriculum_logger("curriculum.dataset_sampler")

    def sample_deterministic(
        self, records: List[DatasetRecord], target_count: int, seed: Optional[int] = None
    ) -> List[DatasetRecord]:
        """
        Samples target_count items deterministically without replacement (or with cycling if count exceeds available).

        Parameters
        ----------
        records : List[DatasetRecord]
            Source list of records.
        target_count : int
            Number of records to sample.
        seed : Optional[int]
            Random seed. Uses config seed if None.

        Returns
        -------
        List[DatasetRecord]
            Sampled records list.
        """
        if not records:
            raise ValueError("Cannot sample from an empty records list.")

        active_seed = seed if seed is not None else self.config.random_seed
        set_seed(active_seed)

        self.logger.info(
            "Sampling %d records deterministically from pool of %d (seed=%d)",
            target_count,
            len(records),
            active_seed,
        )

        # Create copy for shuffling
        shuffled = list(records)
        random.Random(active_seed).shuffle(shuffled)

        sampled: List[DatasetRecord] = []
        cycle_idx = 0
        while len(sampled) < target_count:
            for item in shuffled:
                if len(sampled) >= target_count:
                    break
                if cycle_idx == 0:
                    sampled.append(item)
                else:
                    cycled_item = DatasetRecord(
                        text=f"{item.text} [c{cycle_idx}]",
                        source=item.source,
                        record_id=f"{item.record_id}_c{cycle_idx}",
                        metadata=dict(item.metadata),
                    )
                    sampled.append(cycled_item)
            cycle_idx += 1

        return sampled
