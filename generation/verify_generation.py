"""
================================================================================
generation/verify_generation.py
================================================================================

Project: Anchor-Regularized Model Collapse Prevention
Component: Synthetic Data Generation Pipeline (Generation 1)

Purpose:
    Rigorous verification suite acting as a gatekeeper. Ensures data integrity,
    checks for catastrophic collapse indicators (e.g., duplicate outputs),
    and validates metadata completeness before saving to disk.

Authors: Deepak (Research Lead)
Created: 2026-07-09
License: MIT
================================================================================
"""

import json
from typing import List, Dict, Any
from .generation_config import GenerationConfig
from .utils import setup_logger

log = setup_logger("verify_generation")


def verify_dataset(config: GenerationConfig, records: List[Dict[str, Any]]) -> bool:
    """
    Verify the generated dataset.
    
    Args:
        config (GenerationConfig): Pipeline settings.
        records (List[Dict]): The generated records.
        
    Returns:
        bool: True if verification passes, False otherwise.
    """
    log.info("Starting verification suite...")
    
    # 1. Dataset Size Check
    expected_size = config.max_documents
    actual_size = len(records)
    if actual_size != expected_size:
        log.error("Size mismatch! Expected %d, got %d", expected_size, actual_size)
        return False
    log.info(" [PASS] Dataset size check (%d documents)", actual_size)
    
    # 2. Empty / Short Generations Check
    empty_threshold = 5  # minimum acceptable tokens generated
    short_generations = sum(1 for r in records if r["token_count"] < empty_threshold)
    if short_generations > 0:
        log.warning("Found %d very short generations (< %d tokens)", short_generations, empty_threshold)
        # We don't fail immediately, but flag it. 
        # In a strict setting, we might want to fail. Let's allow a small percentage (e.g. 1%)
        if short_generations > (actual_size * 0.01):
            log.error("Too many short generations. Failing verification.")
            return False
    log.info(" [PASS] Empty output check")
    
    # 3. Duplicate Outputs Check (Entropy Check)
    # We hash the generated text to find exact duplicates.
    unique_texts = set(r["generated_text"] for r in records)
    duplicates = actual_size - len(unique_texts)
    if duplicates > 0:
        log.warning("Found %d duplicate generations.", duplicates)
        if duplicates > (actual_size * 0.05):  # 5% threshold
            log.error("Catastrophic duplication detected. Probable deterministic collapse.")
            return False
    log.info(" [PASS] Duplicate output check")
    
    # 4. Metadata Completeness
    required_keys = ["document_id", "prompt", "generated_text", "token_count", 
                     "generation_parameters", "anchor_version", "generation_timestamp", "metadata"]
    for i, record in enumerate(records):
        for key in required_keys:
            if key not in record:
                log.error("Missing key '%s' in record %d", key, i)
                return False
                
    log.info(" [PASS] Metadata completeness check")
    
    log.info("Verification suite completed successfully.")
    return True
