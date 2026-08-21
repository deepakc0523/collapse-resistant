"""
================================================================================
curriculum/dataset_loader.py
================================================================================

Dataset loader module for human anchor and synthetic data sources.

Reads raw human anchor text/JSONL datasets and synthetic Generation-1 datasets,
normalizing them into structured sample representations for curriculum mixing.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from curriculum.curriculum_config import CurriculumConfig
from curriculum.utils import get_curriculum_logger, compute_text_hash


@dataclass
class DatasetRecord:
    """
    Standardized record item in dataset construction.

    Attributes
    ----------
    text : str
        Text content snippet.
    source : str
        Data origin category ('anchor' or 'synthetic').
    record_id : str
        Unique identifier snippet.
    metadata : Dict[str, Any]
        Optional record-level metadata dictionary.
    """

    text: str
    source: str
    record_id: str
    metadata: Dict[str, Any]


class DatasetLoader:
    """Loads human anchor and synthetic dataset files from disk."""

    def __init__(
        self,
        config: Optional[CurriculumConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or CurriculumConfig()
        self.logger = logger or get_curriculum_logger("curriculum.dataset_loader")

    def load_anchor_dataset(self, limit: Optional[int] = None) -> List[DatasetRecord]:
        """
        Loads canonical human anchor dataset.

        Parameters
        ----------
        limit : Optional[int]
            Maximum number of records to load.

        Returns
        -------
        List[DatasetRecord]
            List of loaded anchor dataset records.
        """
        path = self.config.anchor_dataset_path
        self.logger.info("Loading human anchor dataset from: %s", path)

        records: List[DatasetRecord] = []

        if path.exists() and path.is_file():
            if path.suffix == ".txt":
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line_str = line.strip()
                        if line_str and len(line_str) > 20: # Filter short headings
                            rec_id = f"anc_{compute_text_hash(line_str)}"
                            records.append(
                                DatasetRecord(
                                    text=line_str,
                                    source="anchor",
                                    record_id=rec_id,
                                    metadata={"origin": "clean_wikitext"},
                                )
                            )
                            if limit and len(records) >= limit:
                                break
            elif path.suffix == ".jsonl":
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        text = data.get("text", "")
                        if text:
                            rec_id = data.get("id", f"anc_{compute_text_hash(text)}")
                            records.append(
                                DatasetRecord(
                                    text=text,
                                    source="anchor",
                                    record_id=rec_id,
                                    metadata={"origin": "wikitext_jsonl"},
                                )
                            )
                            if limit and len(records) >= limit:
                                break

        # Fallback generator for isolated module execution or small samples
        if not records:
            self.logger.warning(
                "Anchor dataset path not found or empty. Generating representative canonical anchor fallback records."
            )
            records = self._generate_fallback_records(source="anchor", count=limit or 500)

        self.logger.info("Loaded %d anchor records.", len(records))
        return records

    def load_synthetic_dataset(self, limit: Optional[int] = None) -> List[DatasetRecord]:
        """
        Loads synthetic Generation-1 dataset.

        Parameters
        ----------
        limit : Optional[int]
            Maximum number of records to load.

        Returns
        -------
        List[DatasetRecord]
            List of loaded synthetic dataset records.
        """
        path = self.config.synthetic_dataset_path
        self.logger.info("Loading synthetic Generation-1 dataset from: %s", path)

        records: List[DatasetRecord] = []

        # Check arrow directory or jsonl file
        if path.exists():
            if path.is_file() and path.suffix == ".jsonl":
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        text = data.get("text", "")
                        if text:
                            rec_id = data.get("id", f"syn_{compute_text_hash(text)}")
                            records.append(
                                DatasetRecord(
                                    text=text,
                                    source="synthetic",
                                    record_id=rec_id,
                                    metadata={"origin": "generation_1_jsonl"},
                                )
                            )
                            if limit and len(records) >= limit:
                                break
            elif path.is_dir():
                # Attempt PyArrow load if datasets package is available
                try:
                    from datasets import load_from_disk
                    dataset = load_from_disk(str(path))
                    for item in dataset:
                        text = item.get("text", "") if isinstance(item, dict) else str(item)
                        if text:
                            rec_id = f"syn_{compute_text_hash(text)}"
                            records.append(
                                DatasetRecord(
                                    text=text,
                                    source="synthetic",
                                    record_id=rec_id,
                                    metadata={"origin": "generation_1_arrow"},
                                )
                            )
                            if limit and len(records) >= limit:
                                break
                except Exception as e:
                    self.logger.debug("PyArrow disk load fallback: %s", str(e))

        # Fallback generator for isolated module execution
        if not records:
            self.logger.warning(
                "Synthetic dataset path not found or empty. Generating representative synthetic fallback records."
            )
            records = self._generate_fallback_records(source="synthetic", count=limit or 500)

        self.logger.info("Loaded %d synthetic records.", len(records))
        return records

    def _generate_fallback_records(self, source: str, count: int) -> List[DatasetRecord]:
        """Generates clean representative records for testing."""
        records = []
        prefix = "Human Canonical Anchor Knowledge Sample:" if source == "anchor" else "Model Synthetic Generated Output Sample:"
        for i in range(1, count + 1):
            text = f"{prefix} Recursive language model training sentence #{i} covering natural language statistics, context dependencies, and semantics."
            rec_id = f"{source[:3]}_{compute_text_hash(text)}_{i}"
            records.append(
                DatasetRecord(
                    text=text,
                    source=source,
                    record_id=rec_id,
                    metadata={"fallback": True, "index": i},
                )
            )
        return records
