"""
================================================================================
curriculum/curriculum_scheduler.py
================================================================================

Progressive curriculum scheduler module for the Curriculum Controller.

Arranges mixed dataset samples into an adaptive, progressive ordering from canonical
human anchor foundation to target policy synthetic mixture exposure.
"""

import random
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from curriculum.curriculum_config import CurriculumConfig
from curriculum.dataset_loader import DatasetRecord
from curriculum.dataset_mixer import MixedDatasetPool
from curriculum.policy_loader import ATEPolicyData
from curriculum.utils import get_curriculum_logger, set_seed


@dataclass
class ScheduledCurriculum:
    """
    Container holding ordered dataset sequence and stage boundary metadata.

    Attributes
    ----------
    ordered_records : List[DatasetRecord]
        Complete ordered record sequence for training.
    stage_boundaries : Dict[str, Tuple[int, int]]
        Stage name mapping to (start_idx, end_idx) sample ranges.
    stage_compositions : Dict[str, Dict[str, float]]
        Composition ratios for each stage.
    """

    ordered_records: List[DatasetRecord]
    stage_boundaries: Dict[str, Any]
    stage_compositions: Dict[str, Dict[str, float]]


class CurriculumScheduler:
    """Schedules mixed dataset records into progressive exposure curriculum stages."""

    def __init__(
        self,
        config: Optional[CurriculumConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or CurriculumConfig()
        self.logger = logger or get_curriculum_logger("curriculum.curriculum_scheduler")

    def schedule_curriculum(
        self, pool: MixedDatasetPool, policy_data: ATEPolicyData
    ) -> ScheduledCurriculum:
        """
        Schedules dataset pool into a 3-stage progressive curriculum:
          - Stage 1 (Foundation): 100% Anchor ground truth
          - Stage 2 (Transition): Heavy Anchor / Light Synthetic Interleaved Mix
          - Stage 3 (Advanced Exposure): Target Policy Mix Ratio

        Parameters
        ----------
        pool : MixedDatasetPool
            Sampled dataset pool.
        policy_data : ATEPolicyData
            ATE policy object.

        Returns
        -------
        ScheduledCurriculum
            Ordered dataset sequence with stage annotations.
        """
        set_seed(self.config.random_seed)
        self.logger.info("Scheduling dataset into 3-stage progressive curriculum...")

        anchor_pool = list(pool.anchor_samples)
        synthetic_pool = list(pool.synthetic_samples)

        total_samples = len(anchor_pool) + len(synthetic_pool)

        # Stage size calculation
        n_stage1 = int(round(total_samples * self.config.stage1_ratio))
        n_stage2 = int(round(total_samples * self.config.stage2_ratio))
        n_stage3 = total_samples - n_stage1 - n_stage2

        # --- Stage 1: Foundation (100% Pure Anchor) ---
        stage1_records: List[DatasetRecord] = []
        for _ in range(n_stage1):
            if anchor_pool:
                rec = anchor_pool.pop(0)
                rec.metadata["curriculum_stage"] = "Stage_1_Foundation"
                rec.metadata["stage_id"] = 1
                stage1_records.append(rec)

        # --- Stage 2: Transition (80% Anchor, 20% Synthetic) ---
        stage2_records: List[DatasetRecord] = []
        target_syn_in_stage2 = int(round(n_stage2 * 0.20))
        target_anc_in_stage2 = n_stage2 - target_syn_in_stage2

        syn_count_s2 = 0
        anc_count_s2 = 0

        for i in range(n_stage2):
            # Interleave: add synthetic roughly every 5 samples if available
            want_synthetic = (i % 5 == 4) and (syn_count_s2 < target_syn_in_stage2)
            if want_synthetic and synthetic_pool:
                rec = synthetic_pool.pop(0)
                syn_count_s2 += 1
            elif anchor_pool:
                rec = anchor_pool.pop(0)
                anc_count_s2 += 1
            elif synthetic_pool:
                rec = synthetic_pool.pop(0)
                syn_count_s2 += 1
            else:
                break

            rec.metadata["curriculum_stage"] = "Stage_2_Transition"
            rec.metadata["stage_id"] = 2
            stage2_records.append(rec)

        # --- Stage 3: Advanced Exposure (Remaining Anchor + Remaining Synthetic) ---
        stage3_records: List[DatasetRecord] = []
        remaining_pool = anchor_pool + synthetic_pool
        # Shuffle remaining for deterministic stage 3 mix
        rng = random.Random(self.config.random_seed + 2)
        rng.shuffle(remaining_pool)

        for rec in remaining_pool:
            rec.metadata["curriculum_stage"] = "Stage_3_Advanced"
            rec.metadata["stage_id"] = 3
            stage3_records.append(rec)

        ordered_all = stage1_records + stage2_records + stage3_records

        # Calculate stage metadata
        idx1_end = len(stage1_records)
        idx2_end = idx1_end + len(stage2_records)
        idx3_end = len(ordered_all)

        stage_boundaries = {
            "Stage_1_Foundation": [0, idx1_end],
            "Stage_2_Transition": [idx1_end, idx2_end],
            "Stage_3_Advanced": [idx2_end, idx3_end],
        }

        stage_compositions = {
            "Stage_1_Foundation": {
                "anchor_ratio": 1.0,
                "synthetic_ratio": 0.0,
                "count": len(stage1_records),
            },
            "Stage_2_Transition": {
                "anchor_ratio": round(anc_count_s2 / max(1, len(stage2_records)), 4),
                "synthetic_ratio": round(syn_count_s2 / max(1, len(stage2_records)), 4),
                "count": len(stage2_records),
            },
            "Stage_3_Advanced": {
                "anchor_ratio": round(
                    sum(1 for r in stage3_records if r.source == "anchor") / max(1, len(stage3_records)), 4
                ),
                "synthetic_ratio": round(
                    sum(1 for r in stage3_records if r.source == "synthetic") / max(1, len(stage3_records)), 4
                ),
                "count": len(stage3_records),
            },
        }

        scheduled = ScheduledCurriculum(
            ordered_records=ordered_all,
            stage_boundaries=stage_boundaries,
            stage_compositions=stage_compositions,
        )

        self.logger.info(
            "Curriculum scheduling complete. Total: %d samples [Stage 1: %d, Stage 2: %d, Stage 3: %d]",
            len(ordered_all),
            len(stage1_records),
            len(stage2_records),
            len(stage3_records),
        )

        return scheduled
