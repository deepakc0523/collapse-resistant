"""
================================================================================
generation/statistics.py
================================================================================

Project: Anchor-Regularized Model Collapse Prevention
Component: Synthetic Data Generation Pipeline (Generation 1)

Purpose:
    Computes summary statistics for the generated dataset, providing insights
    into length distributions, vocabulary diversity, and throughput.

Authors: Deepak (Research Lead)
Created: 2026-07-09
License: MIT
================================================================================
"""

import json
from typing import List, Dict, Any
from .generation_config import GenerationConfig
from .utils import setup_logger

log = setup_logger("statistics")


def generate_statistics(
    config: GenerationConfig, 
    records: List[Dict[str, Any]], 
    total_time: float
) -> Dict[str, Any]:
    """
    Compute dataset statistics.
    
    Args:
        config (GenerationConfig): Pipeline settings.
        records (List[Dict]): The generated records.
        total_time (float): Time taken to generate the dataset.
        
    Returns:
        Dict: A dictionary of computed statistics.
    """
    log.info("Computing dataset statistics...")
    
    token_counts = [r["token_count"] for r in records]
    total_tokens = sum(token_counts)
    
    max_len = max(token_counts) if token_counts else 0
    min_len = min(token_counts) if token_counts else 0
    avg_len = total_tokens / len(token_counts) if token_counts else 0
    
    # Very basic vocabulary diversity estimation (word level split)
    # Note: For strict ML purposes we'd use the tokenizer, but this provides a fast proxy.
    unique_words = set()
    for r in records:
        words = r["generated_text"].split()
        unique_words.update(words)
        
    vocab_size = len(unique_words)
    total_words_approx = sum(len(r["generated_text"].split()) for r in records)
    unique_token_ratio = vocab_size / total_words_approx if total_words_approx > 0 else 0
    
    gen_speed_docs = len(records) / total_time if total_time > 0 else 0
    gen_speed_tokens = total_tokens / total_time if total_time > 0 else 0
    
    stats = {
        "documents_generated": len(records),
        "total_tokens_generated": total_tokens,
        "average_document_length": avg_len,
        "maximum_length": max_len,
        "minimum_length": min_len,
        "vocabulary_size_approx": vocab_size,
        "unique_token_ratio": unique_token_ratio,
        "generation_time_seconds": total_time,
        "generation_speed_docs_per_sec": gen_speed_docs,
        "generation_speed_tokens_per_sec": gen_speed_tokens,
        "hardware_device": config.device
    }
    
    log.info("Stats Computed:")
    for k, v in stats.items():
        if isinstance(v, float):
            log.info("  %s: %.4f", k, v)
        else:
            log.info("  %s: %s", k, v)
            
    # Save the report
    report_path = config.output_dir.parent / f"{config.output_dir.name}_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
        
    log.info("Statistics report saved to %s", report_path)
    return stats
