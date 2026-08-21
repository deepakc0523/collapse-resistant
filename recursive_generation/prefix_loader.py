"""
================================================================================
recursive_generation/prefix_loader.py
================================================================================

Loads the canonical human prefix dataset (clean_wikitext.txt) for prompting
the Generation-1 student during Generation-2 synthesis.

The SAME prefixes used in Generation-1 are reused here intentionally.
This ensures generational continuity — the recursive learner is tested
on the same anchors, not fresh random prompts.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from recursive_generation.generation_config import GenerationConfig
from recursive_generation.utils import get_generation_logger


@dataclass
class PrefixRecord:
    """
    Single prefix record extracted from the anchor text corpus.

    Attributes
    ----------
    text : str
        Raw prefix text.
    index : int
        Ordinal index in the full prefix list.
    char_length : int
        Character length of the prefix text.
    """

    text: str
    index: int
    char_length: int


class PrefixLoader:
    """
    Loads and prepares human anchor text prefixes for Generation-2 prompting.
    """

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or GenerationConfig()
        self.logger = logger or get_generation_logger("recursive_generation.prefix_loader")

    def load_prefixes(
        self,
        path: Optional[Path] = None,
        min_length: int = 30,
        limit: Optional[int] = None,
    ) -> List[PrefixRecord]:
        """
        Reads clean_wikitext.txt and extracts non-trivial lines as generation prompts.

        Parameters
        ----------
        path : Optional[Path]
            Override path. Uses config path if None.
        min_length : int
            Minimum character length for a valid prefix. Filters headings and noise.
        limit : Optional[int]
            Maximum number of prefixes to load. None = all.

        Returns
        -------
        List[PrefixRecord]
            Ordered list of prefix records.
        """
        target_path = path or self.config.prefix_dataset_path
        self.logger.info("Loading prefix dataset from: %s", target_path)

        if not target_path.exists():
            raise FileNotFoundError(
                f"Prefix dataset not found: {target_path}. "
                "Ensure data preprocessing has been completed."
            )

        records: List[PrefixRecord] = []
        seen: set = set()

        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip()
                if len(line) < min_length:
                    continue
                if line in seen:
                    continue
                seen.add(line)

                records.append(
                    PrefixRecord(
                        text=line,
                        index=len(records),
                        char_length=len(line),
                    )
                )

                if limit is not None and len(records) >= limit:
                    break

        self.logger.info(
            "Loaded %d valid prefixes from corpus (min_length=%d).",
            len(records),
            min_length,
        )
        return records
