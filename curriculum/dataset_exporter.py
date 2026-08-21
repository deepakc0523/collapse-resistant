"""
================================================================================
curriculum/dataset_exporter.py
================================================================================

Dataset exporter module for the Curriculum Controller.

Serializes constructed Generation-2 training split (train.jsonl), validation split
(validation.jsonl), and metadata (metadata.json) to curriculum_out/generation_2/.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from curriculum.curriculum_config import CurriculumConfig
from curriculum.dataset_loader import DatasetRecord
from curriculum.utils import get_curriculum_logger


class DatasetExporter:
    """Exports train.jsonl, validation.jsonl, and metadata.json to disk."""

    def __init__(
        self,
        config: Optional[CurriculumConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or CurriculumConfig()
        self.logger = logger or get_curriculum_logger("curriculum.dataset_exporter")

    def export_dataset(
        self,
        records: List[DatasetRecord],
        metadata_payload: Dict[str, Any],
    ) -> Tuple[Path, Path, Path]:
        """
        Splits records into train/val and exports all files to disk.

        Parameters
        ----------
        records : List[DatasetRecord]
            Ordered dataset records.
        metadata_payload : Dict[str, Any]
            Metadata dictionary.

        Returns
        -------
        Tuple[Path, Path, Path]
            Paths to (train.jsonl, validation.jsonl, metadata.json).
        """
        self.logger.info("Exporting Generation-2 dataset artifacts...")

        split_idx = int(round(len(records) * self.config.train_val_split))
        train_records = records[:split_idx]
        val_records = records[split_idx:]

        train_path = self.config.train_jsonl_path
        val_path = self.config.val_jsonl_path
        meta_path = self.config.metadata_json_path

        # Export train.jsonl
        with open(train_path, "w", encoding="utf-8") as f:
            for i, rec in enumerate(train_records):
                item = {
                    "id": rec.record_id,
                    "text": rec.text,
                    "source": rec.source,
                    "sample_index": i,
                    "metadata": rec.metadata,
                }
                f.write(json.dumps(item) + "\n")

        # Export validation.jsonl
        with open(val_path, "w", encoding="utf-8") as f:
            for i, rec in enumerate(val_records):
                item = {
                    "id": rec.record_id,
                    "text": rec.text,
                    "source": rec.source,
                    "sample_index": i,
                    "metadata": rec.metadata,
                }
                f.write(json.dumps(item) + "\n")

        # Export metadata.json
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=4)

        self.logger.info("Exported %d train samples to: %s", len(train_records), train_path)
        self.logger.info("Exported %d val samples to: %s", len(val_records), val_path)
        self.logger.info("Exported metadata to: %s", meta_path)

        return train_path, val_path, meta_path
