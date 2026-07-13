"""
================================================================================
probe/prompt_loader.py
================================================================================

Loads, samples, and tokenizes textual prompts for representation analysis.
Supports wikitext files (paragraph chunks) as well as line-by-line custom lists.
"""

import random
import logging
from pathlib import Path
from typing import List, Tuple
from transformers import PreTrainedTokenizer

logger = logging.getLogger("probe.prompt_loader")

def load_prompts(
    file_path: Path,
    tokenizer: PreTrainedTokenizer,
    max_prompts: int = 100,
    min_tokens: int = 32,
    max_tokens: int = 64,
    seed: int = 42
) -> List[str]:
    """
    Loads prompt strings from clean_wikitext.txt or custom files.
    If the file is a wikitext file (determined by directory or content), 
    we split by double newline and sample prefixes. Otherwise, we treat each non-empty
    line as a distinct prompt.
    
    Args:
        file_path: Path to wikitext or custom prompt file.
        tokenizer: Model tokenizer used to measure token lengths accurately.
        max_prompts: Maximum number of prompts to load/sample.
        min_tokens: Minimum prompt length in tokens (for wikitext sampling).
        max_tokens: Maximum prompt length in tokens (for wikitext sampling).
        seed: Random seed for deterministic sampling.
        
    Returns:
        List[str]: List of selected prompt strings.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found at: {file_path}")

    logger.info("Loading prompts from: %s", file_path)
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Determine if it's wikitext formatted (paragraphs split by double newline)
    is_wikitext = "wikitext" in file_path.name.lower() or "\n\n" in content

    if is_wikitext:
        logger.info("Parsing file as WikiText (double-newline separated documents)")
        documents = [doc.strip() for doc in content.split("\n\n") if len(doc.strip()) > 100]
        
        if not documents:
            # Fallback to single newline split if wikitext split resulted in nothing
            documents = [line.strip() for line in content.split("\n") if len(line.strip()) > 20]
            
        logger.info("Found %d raw documents. Sampling prefixes...", len(documents))
        
        # Deterministic sampling
        random.seed(seed)
        if len(documents) > max_prompts:
            sampled_docs = random.sample(documents, max_prompts)
        else:
            sampled_docs = documents
            
        prompts = []
        for doc in sampled_docs:
            tokens = tokenizer.encode(doc, add_special_tokens=False)
            if len(tokens) < min_tokens:
                # If too short, just take what we have
                prompt_len = len(tokens)
            else:
                prompt_len = random.randint(min_tokens, min(max_tokens, len(tokens)))
            
            prompt_tokens = tokens[:prompt_len]
            prompt_text = tokenizer.decode(prompt_tokens, clean_up_tokenization_spaces=True)
            prompts.append(prompt_text)
            
    else:
        logger.info("Parsing file as plain text (line-by-line prompts)")
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        
        random.seed(seed)
        if len(lines) > max_prompts:
            prompts = random.sample(lines, max_prompts)
        else:
            prompts = lines

    logger.info("Successfully loaded and processed %d prompts.", len(prompts))
    return prompts
