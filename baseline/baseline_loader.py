"""
================================================================================
baseline/baseline_loader.py
================================================================================

Loads real Generation-2 synthetic dataset for the Student-2 control experiment.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from baseline.baseline_config import BaselineConfig
from baseline.utils import get_baseline_logger, compute_text_hash


@dataclass
class BaselineRecord:
    """
    Standardized dataset record container.

    Attributes
    ----------
    text : str
        Text content for training.
    source : str
        Source identifier ("synthetic").
    record_id : str
        Unique string identifier.
    metadata : Dict[str, Any]
        Record-level provenance metadata dictionary.
    """

    text: str
    source: str
    record_id: str
    metadata: Dict[str, Any]


class BaselineLoader:
    """Loads raw Generation-2 synthetic records from disk."""

    def __init__(
        self,
        config: Optional[BaselineConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or BaselineConfig()
        self.logger = logger or get_baseline_logger("baseline.baseline_loader")

    def load_synthetic_dataset(
        self,
        allow_fallback: bool = False,
        required_records: int = 1000,
    ) -> List[BaselineRecord]:
        """
        Loads real Generation-2 synthetic JSONL file.

        Parameters
        ----------
        allow_fallback : bool
            If False, fails loudly when file is missing, empty, malformed, or has fewer than required_records.
        required_records : int
            Expected record count for the control experiment (1,000).

        Returns
        -------
        List[BaselineRecord]
            Loaded synthetic records.
        """
        path = self.config.synthetic_dataset_path
        self.logger.info("Loading real Generation-2 synthetic dataset for baseline from: %s", path)

        if not path.exists():
            if not allow_fallback:
                raise FileNotFoundError(
                    f"Real Generation-2 synthetic dataset file not found at required path: {path}. "
                    "Ensure generation_2 synthesis has completed."
                )
            self.logger.warning(
                "Synthetic dataset path not found (%s). Generating fallback records.", path
            )
            return self._generate_fallback_records(count=required_records)

        records: List[BaselineRecord] = []

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

                    # Use synthetic continuation text, fallback to full_text or text
                    continuation = (
                        data.get("generated_continuation")
                        or data.get("full_text")
                        or data.get("text")
                        or ""
                    ).strip()

                    if not continuation:
                        continue

                    prompt_idx = data.get("_prompt_index", line_num - 1)
                    rec_id = f"base_gen2_{prompt_idx:04d}_{compute_text_hash(continuation)[:8]}"

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
                        BaselineRecord(
                            text=continuation,
                            source="synthetic",
                            record_id=rec_id,
                            metadata=metadata,
                        )
                    )

        if not records:
            if not allow_fallback:
                raise ValueError(
                    f"Real Generation-2 synthetic dataset at {path} is empty or contains no valid records. Failing loudly."
                )
            records = self._generate_fallback_records(count=required_records)

        if not allow_fallback and len(records) < required_records:
            raise ValueError(
                f"Real Generation-2 synthetic dataset requires at least {required_records} records, but only loaded {len(records)} from {path}."
            )

        self.logger.info("Successfully loaded %d real Generation-2 synthetic records for baseline.", len(records))
        return records

    def _generate_fallback_records(self, count: int) -> List[BaselineRecord]:
        """Generates dummy records only when fallback is explicitly allowed during isolation testing."""
        records = []
        for i in range(1, count + 1):
            text = f"Model Synthetic Generated Output Sample sentence #{i} for baseline control experiment."
            rec_id = f"base_syn_{compute_text_hash(text)}_{i}"
            records.append(
                BaselineRecord(
                    text=text,
                    source="synthetic",
                    record_id=rec_id,
                    metadata={
                        "fallback": True,
                        "generation": 2,
                        "parent_student": "generation_1",
                        "index": i,
                    },
                )
            )
        return records
