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
import random
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
        self.available_prefix_count: int = 0
        self.selected_prefix_count: int = 0

    def load_prefixes(
        self,
        path: Optional[Path] = None,
        min_length: int = 30,
        max_prompts: Optional[int] = None,
        seed: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[PrefixRecord]:
        """
        Reads clean_wikitext.txt, extracts valid non-trivial lines, and deterministically
        samples up to max_prompts prefixes while preserving relative corpus order.

        Parameters
        ----------
        path : Optional[Path]
            Override path. Uses config path if None.
        min_length : int
            Minimum character length for a valid prefix. Filters headings and noise.
        max_prompts : Optional[int]
            Maximum number of prefixes to sample deterministically. None = use config.
        seed : Optional[int]
            Random seed for deterministic sampling. None = use config.
        limit : Optional[int]
            Legacy alias for max_prompts.

        Returns
        -------
        List[PrefixRecord]
            Ordered list of prefix records after deterministic sampling.
        """
        target_path = path or self.config.prefix_dataset_path
        target_max_prompts = (
            max_prompts
            if max_prompts is not None
            else (limit if limit is not None else self.config.max_prompts)
        )
        target_seed = seed if seed is not None else self.config.random_seed

        self.logger.info("Loading prefix dataset from: %s", target_path)

        if not target_path.exists():
            raise FileNotFoundError(
                f"Prefix dataset not found: {target_path}. "
                "Ensure data preprocessing has been completed."
            )

        all_valid_lines: List[str] = []
        seen: set = set()

        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip()
                if len(line) < min_length:
                    continue
                if line in seen:
                    continue
                seen.add(line)
                all_valid_lines.append(line)

        self.available_prefix_count = len(all_valid_lines)

        if target_max_prompts is not None and self.available_prefix_count > target_max_prompts:
            rng = random.Random(target_seed)
            sampled_indices = sorted(rng.sample(range(self.available_prefix_count), target_max_prompts))
            selected_lines = [all_valid_lines[i] for i in sampled_indices]
        else:
            selected_lines = all_valid_lines

        records: List[PrefixRecord] = [
            PrefixRecord(
                text=line,
                index=i,
                char_length=len(line),
            )
            for i, line in enumerate(selected_lines)
        ]

        self.selected_prefix_count = len(records)

        self.logger.info(
            "Loaded %d valid prefixes from corpus (total available=%d, max_prompts=%s, seed=%d).",
            self.selected_prefix_count,
            self.available_prefix_count,
            str(target_max_prompts),
            target_seed,
        )
        return records
