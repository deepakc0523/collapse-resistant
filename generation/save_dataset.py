"""
================================================================================
generation/save_dataset.py
================================================================================

Project: Anchor-Regularized Model Collapse Prevention
Component: Synthetic Data Generation Pipeline (Generation 1)

Purpose:
    Handles storing the generated Python dictionary records into a highly
    optimized Apache Arrow format via HuggingFace Datasets.

Authors: Deepak (Research Lead)
Created: 2026-07-09
License: MIT
================================================================================
"""

from typing import List, Dict, Any
import datasets
from .generation_config import GenerationConfig
from .utils import setup_logger

log = setup_logger("save_dataset")


def save_to_arrow(config: GenerationConfig, records: List[Dict[str, Any]]) -> None:
    """
    Convert dictionary records to a HuggingFace Dataset and save to disk.
    
    Args:
        config (GenerationConfig): Pipeline settings.
        records (List[Dict]): The generated records.
    """
    log.info("Converting %d records to HuggingFace Arrow format...", len(records))
    
    # HuggingFace Datasets seamlessly converts lists of dicts to Arrow tables
    dataset = datasets.Dataset.from_list(records)
    
    output_path = config.output_dir
    log.info("Saving dataset to %s", output_path)
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to disk
    dataset.save_to_disk(str(output_path))
    
    log.info("Dataset successfully saved. Size on disk will be highly compressed.")
