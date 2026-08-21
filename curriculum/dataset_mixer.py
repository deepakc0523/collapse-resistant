"""
================================================================================
curriculum/dataset_mixer.py
================================================================================

Dataset mixer module for the Curriculum Controller.

Calculates target sample counts based on ATE policy mix ratios (synthetic_ratio,
anchor_ratio) and samples anchor and synthetic records proportionally.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

from curriculum.curriculum_config import CurriculumConfig
from curriculum.policy_loader import ATEPolicyData
from curriculum.dataset_loader import DatasetRecord
from curriculum.dataset_sampler import DatasetSampler
from curriculum.utils import get_curriculum_logger


@dataclass
class MixedDatasetPool:
    """
    Container holding sampled anchor and synthetic records prior to curriculum scheduling.

    Attributes
    ----------
    anchor_samples : List[DatasetRecord]
        Sampled anchor records pool.
    synthetic_samples : List[DatasetRecord]
        Sampled synthetic records pool.
    target_anchor_count : int
        Calculated target count for anchor samples.
    target_synthetic_count : int
        Calculated target count for synthetic samples.
    actual_synthetic_ratio : float
        Achieved synthetic ratio.
    actual_anchor_ratio : float
        Achieved anchor ratio.
    """

    anchor_samples: List[DatasetRecord]
    synthetic_samples: List[DatasetRecord]
    target_anchor_count: int
    target_synthetic_count: int
    actual_synthetic_ratio: float
    actual_anchor_ratio: float


class DatasetMixer:
    """Mixes anchor and synthetic datasets according to derived ATE policy ratios."""

    def __init__(
        self,
        config: Optional[CurriculumConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or CurriculumConfig()
        self.logger = logger or get_curriculum_logger("curriculum.dataset_mixer")
        self.sampler = DatasetSampler(config=self.config, logger=self.logger)

    def mix_datasets(
        self,
        anchor_pool: List[DatasetRecord],
        synthetic_pool: List[DatasetRecord],
        policy_data: ATEPolicyData,
        total_count: Optional[int] = None,
    ) -> MixedDatasetPool:
        """
        Calculates exact sample split matching policy ratios and samples records.

        Parameters
        ----------
        anchor_pool : List[DatasetRecord]
            Source pool of anchor records.
        synthetic_pool : List[DatasetRecord]
            Source pool of synthetic records.
        policy_data : ATEPolicyData
            Parsed ATE policy object containing synthetic_ratio and anchor_ratio.
        total_count : Optional[int]
            Target total dataset size. Uses config size if None.

        Returns
        -------
        MixedDatasetPool
            Structured container of sampled records.
        """
        n_total = total_count or self.config.total_dataset_size
        syn_ratio = policy_data.synthetic_ratio
        anc_ratio = policy_data.anchor_ratio

        n_synthetic = int(round(n_total * syn_ratio))
        n_anchor = n_total - n_synthetic

        self.logger.info(
            "Mixing datasets for target size %d (Policy Ratios: Synthetic=%.4f [%d], Anchor=%.4f [%d])",
            n_total,
            syn_ratio,
            n_synthetic,
            anc_ratio,
            n_anchor,
        )

        sampled_anchor = self.sampler.sample_deterministic(
            anchor_pool, target_count=n_anchor, seed=self.config.random_seed
        )

        sampled_synthetic = self.sampler.sample_deterministic(
            synthetic_pool, target_count=n_synthetic, seed=self.config.random_seed + 1
        )

        actual_syn_ratio = len(sampled_synthetic) / n_total if n_total > 0 else 0.0
        actual_anc_ratio = len(sampled_anchor) / n_total if n_total > 0 else 0.0

        pool = MixedDatasetPool(
            anchor_samples=sampled_anchor,
            synthetic_samples=sampled_synthetic,
            target_anchor_count=n_anchor,
            target_synthetic_count=n_synthetic,
            actual_synthetic_ratio=actual_syn_ratio,
            actual_anchor_ratio=actual_anc_ratio,
        )

        self.logger.info(
            "Dataset mixing complete. Anchor Count: %d, Synthetic Count: %d",
            len(sampled_anchor),
            len(sampled_synthetic),
        )

        return pool
