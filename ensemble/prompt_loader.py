"""
================================================================================
ensemble/prompt_loader.py
================================================================================

Loads, samples, and tokenises textual prompts for uncertainty evaluation.

This implementation mirrors probe/prompt_loader.py in logic and interface.
It supports both WikiText-formatted files (paragraph chunks separated by
double newlines) and plain line-by-line prompt files.

Sampling is deterministic given a fixed random seed — identical seed values
produce identical prompt sets across runs.
"""

import random
import logging
from pathlib import Path
from typing import List

from transformers import PreTrainedTokenizer

logger = logging.getLogger("ensemble.prompt_loader")


def load_prompts(
    file_path: Path,
    tokenizer: PreTrainedTokenizer,
    max_prompts: int = 100,
    min_tokens: int = 32,
    max_tokens: int = 64,
    seed: int = 42,
) -> List[str]:
    """
    Loads prompt strings from a WikiText or plain-text file.

    For WikiText files (detected by filename or double-newline presence):
      - Splits the file by double newlines into paragraph documents.
      - Samples up to max_prompts documents deterministically.
      - Truncates each document to a random length in [min_tokens, max_tokens].

    For plain-text files:
      - Treats each non-empty line as an independent prompt.
      - Samples up to max_prompts lines deterministically.

    Args:
        file_path:   Path to the prompt source file.
        tokenizer:   Model tokenizer used for accurate token-length measurement.
        max_prompts: Maximum number of prompts to return.
        min_tokens:  Minimum acceptable prompt length in tokens (wikitext only).
        max_tokens:  Maximum acceptable prompt length in tokens (wikitext only).
        seed:        Random seed for reproducible sampling.

    Returns:
        List[str]: Sampled and tokenised prompt strings.

    Raises:
        FileNotFoundError: If file_path does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Prompt source file not found at: {file_path}\n"
            f"Ensure the data pipeline (cleaning / tokenisation) has been run."
        )

    logger.info("Loading prompts from: %s", file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Detect WikiText format by filename or content structure
    is_wikitext = "wikitext" in file_path.name.lower() or "\n\n" in content

    if is_wikitext:
        logger.info("Detected WikiText format (double-newline paragraph separation).")
        documents = [doc.strip() for doc in content.split("\n\n") if len(doc.strip()) > 100]

        if not documents:
            # Fallback: single-newline split for degenerate files
            logger.warning(
                "Double-newline split yielded no documents. Falling back to single-newline split."
            )
            documents = [line.strip() for line in content.split("\n") if len(line.strip()) > 20]

        logger.info("Found %d raw paragraphs. Sampling up to %d...", len(documents), max_prompts)

        random.seed(seed)
        sampled_docs = (
            random.sample(documents, max_prompts) if len(documents) > max_prompts else documents
        )

        prompts: List[str] = []
        for doc in sampled_docs:
            tokens = tokenizer.encode(doc, add_special_tokens=False)
            if len(tokens) < min_tokens:
                # Paragraph too short — use all available tokens
                prompt_len = len(tokens)
            else:
                prompt_len = random.randint(min_tokens, min(max_tokens, len(tokens)))

            prompt_tokens = tokens[:prompt_len]
            prompt_text = tokenizer.decode(prompt_tokens, clean_up_tokenization_spaces=True)
            prompts.append(prompt_text)

    else:
        logger.info("Detected plain-text format (line-by-line prompts).")
        lines = [line.strip() for line in content.split("\n") if line.strip()]

        random.seed(seed)
        prompts = random.sample(lines, max_prompts) if len(lines) > max_prompts else lines

    logger.info("Successfully loaded and processed %d prompts.", len(prompts))
    return prompts
