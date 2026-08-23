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

    def load_anchor_dataset(
        self, limit: Optional[int] = None, allow_fallback: bool = False
    ) -> List[DatasetRecord]:
        """
        Loads canonical human anchor dataset.

        Parameters
        ----------
        limit : Optional[int]
            Maximum number of records to load.
        allow_fallback : bool
            Whether to generate synthetic/anchor fallback records if missing. Default False.

        Returns
        -------
        List[DatasetRecord]
            List of loaded anchor dataset records.
        """
        path = self.config.anchor_dataset_path
        self.logger.info("Loading human anchor dataset from: %s", path)

        if not path.exists() and not allow_fallback:
            raise FileNotFoundError(
                f"Human anchor dataset file not found at required path: {path}. "
                "Ensure clean_wikitext.txt exists."
            )

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
                            if limit and len(records) >= limit and allow_fallback:
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
                            if limit and len(records) >= limit and allow_fallback:
                                break

        if not records:
            if not allow_fallback:
                raise ValueError(
                    f"Human anchor dataset at {path} is empty or produced no valid records. Failing loudly."
                )
            self.logger.warning(
                "Anchor dataset path not found or empty. Generating representative canonical anchor fallback records."
            )
            records = self._generate_fallback_records(source="anchor", count=limit or 500)

        self.logger.info("Loaded %d anchor records.", len(records))
        return records

    def load_synthetic_dataset(
        self,
        limit: Optional[int] = None,
        allow_fallback: bool = False,
        required_records: int = 1000,
    ) -> List[DatasetRecord]:
        """
        Loads real Generation-2 synthetic dataset (generation_2_synthetic.jsonl).

        Parameters
        ----------
        limit : Optional[int]
            Maximum number of records to load.
        allow_fallback : bool
            If False, fails loudly when file is missing, malformed, empty, or has fewer than required_records.
        required_records : int
            Expected minimum record count for the real experiment (1,000).

        Returns
        -------
        List[DatasetRecord]
            List of loaded synthetic dataset records.
        """
        path = self.config.synthetic_dataset_path
        self.logger.info("Loading real Generation-2 synthetic dataset from: %s", path)

        if not path.exists():
            if not allow_fallback:
                raise FileNotFoundError(
                    f"REAL Generation-2 synthetic dataset file not found at required path: {path}. "
                    "Ensure generation_2 synthesis has completed."
                )
            self.logger.warning(
                "Synthetic dataset path not found (%s). Generating fallback records.", path
            )
            return self._generate_fallback_records(source="synthetic", count=limit or 500)

        records: List[DatasetRecord] = []

        if path.is_file() and path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f, 1):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        data = json.loads(line_str)
                    except json.JSONDecodeError as e:
                        if not allow_fallback:
                            raise ValueError(
                                f"Malformed JSON on line {line_num} of real synthetic dataset {path}: {e}"
                            )
                        continue

                    # Use generated synthetic continuation, fallback to full_text or text
                    continuation = (
                        data.get("generated_continuation")
                        or data.get("full_text")
                        or data.get("text")
                        or ""
                    ).strip()

                    if not continuation:
                        continue

                    prompt_idx = data.get("_prompt_index", line_num - 1)
                    rec_id = f"syn_gen2_{prompt_idx:04d}_{compute_text_hash(continuation)[:8]}"

                    metadata = {
                        "origin": "generation_2_synthetic_jsonl",
                        "generation": data.get("generation", 2),
                        "parent_student": data.get("parent_student", "generation_1"),
                        "prompt": data.get("prompt", ""),
                        "generated_continuation": data.get("generated_continuation", ""),
                        "full_text": data.get("full_text", ""),
                        "temperature": data.get("temperature"),
                        "top_p": data.get("top_p"),
                        "top_k": data.get("top_k"),
                        "repetition_penalty": data.get("repetition_penalty"),
                        "max_new_tokens": data.get("max_new_tokens"),
                        "seed": data.get("seed"),
                        "_prompt_index": prompt_idx,
                    }

                    records.append(
                        DatasetRecord(
                            text=continuation,
                            source="synthetic",
                            record_id=rec_id,
                            metadata=metadata,
                        )
                    )
                    if limit and len(records) >= limit and allow_fallback:
                        break

        elif path.is_dir():
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
                                metadata={"origin": "generation_2_arrow"},
                            )
                        )
                        if limit and len(records) >= limit and allow_fallback:
                            break
            except Exception as e:
                self.logger.debug("PyArrow disk load fallback: %s", str(e))

        if not records:
            if not allow_fallback:
                raise ValueError(
                    f"Real Generation-2 synthetic dataset at {path} is empty or contains no valid records. Failing loudly."
                )
            self.logger.warning(
                "Synthetic dataset path not found or empty. Generating representative synthetic fallback records."
            )
            records = self._generate_fallback_records(source="synthetic", count=limit or 500)

        if not allow_fallback and len(records) < required_records:
            raise ValueError(
                f"Real Generation-2 synthetic dataset requires at least {required_records} records, but only loaded {len(records)} from {path}."
            )

        self.logger.info("Loaded %d real synthetic records.", len(records))
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
