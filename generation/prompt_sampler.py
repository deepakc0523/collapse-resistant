"""
================================================================================
generation/prompt_sampler.py
================================================================================

Project: Anchor-Regularized Model Collapse Prevention
Component: Synthetic Data Generation Pipeline (Generation 1)

Purpose:
    Loads the original WikiText-103 dataset, samples 10,000 documents, and
    extracts the first N tokens (32-64) from each to serve as realistic 
    distribution anchors (prompts) for the generator.

Authors: Deepak (Research Lead)
Created: 2026-07-09
License: MIT
================================================================================
"""

import random
from typing import List, Tuple
from transformers import PreTrainedTokenizerBase

from .generation_config import GenerationConfig
from .utils import setup_logger

log = setup_logger("prompt_sampler")


def sample_prompts(
    config: GenerationConfig, 
    tokenizer: PreTrainedTokenizerBase
) -> List[Tuple[str, str]]:
    """
    Sample documents and extract prompts.
    
    Args:
        config (GenerationConfig): The pipeline configuration.
        tokenizer (PreTrainedTokenizerBase): The model's tokenizer.
        
    Returns:
        List[Tuple[str, str]]: A list of tuples containing (original_doc_id, prompt_text).
    """
    log.info("Loading clean_wikitext.txt from: %s", config.dataset_source)
    
    if not config.dataset_source.exists():
        raise FileNotFoundError(f"Dataset source not found: {config.dataset_source}")
        
    # Load all valid documents (basic split by double newline as per WikiText convention)
    with open(config.dataset_source, "r", encoding="utf-8") as f:
        content = f.read()
        
    documents = [doc.strip() for doc in content.split("\n\n") if len(doc.strip()) > 100]
    log.info("Found %d valid documents in corpus.", len(documents))
    
    if len(documents) < config.max_documents:
        log.warning(
            "Requested %d documents, but only %d available. Using all available.",
            config.max_documents, len(documents)
        )
        sampled_docs = documents
    else:
        random.seed(config.random_seed)
        sampled_docs = random.sample(documents, config.max_documents)
        
    log.info("Sampled %d documents. Extracting prefixes...", len(sampled_docs))
    
    prompts = []
    for i, doc in enumerate(sampled_docs):
        # We need the exact tokens, so we tokenize, truncate, and decode back
        doc_id = f"wiki103_doc_{i}"
        
        # Tokenize without special tokens for pure text extraction
        tokens = tokenizer.encode(doc, add_special_tokens=False)
        
        # Determine prompt length for this document
        prompt_len = random.randint(config.prompt_min_tokens, config.prompt_max_tokens)
        
        # Ensure we don't take the whole document if it's very short
        prompt_len = min(prompt_len, max(1, len(tokens) // 2))
        
        prompt_tokens = tokens[:prompt_len]
        prompt_text = tokenizer.decode(prompt_tokens, clean_up_tokenization_spaces=True)
        
        prompts.append((doc_id, prompt_text))
        
    log.info("Successfully extracted %d prompts.", len(prompts))
    return prompts
