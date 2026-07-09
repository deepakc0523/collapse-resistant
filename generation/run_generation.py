"""
================================================================================
generation/run_generation.py
================================================================================

Project: Anchor-Regularized Model Collapse Prevention
Component: Synthetic Data Generation Pipeline (Generation 1)

Purpose:
    Main entry point for generating the synthetic dataset.
    Orchestrates prompt sampling, model loading, text generation, verification,
    statistical analysis, and storage into the Apache Arrow format.

Authors: Deepak (Research Lead)
Created: 2026-07-09
License: MIT
================================================================================
"""

import sys
import traceback
from .generation_config import GenerationConfig
from .utils import setup_logger
from .prompt_sampler import sample_prompts
from .generator import load_anchor_model, generate_synthetic_dataset
from .verify_generation import verify_dataset
from .statistics import generate_statistics
from .save_dataset import save_to_arrow

def main():
    # Initialize Config
    config = GenerationConfig()
    
    # Initialize Logger
    log_file = config.project_root / "checkpoints" / "generation_logs" / "generation_run.log"
    log = setup_logger("run_generation", log_file=log_file)
    
    log.info("============================================================")
    log.info("  Generation 1 Synthetic Data Pipeline Started")
    log.info("============================================================")
    log.info("Mode: %s", "Google Colab (GPU)" if config.is_colab else "Local Development (CPU)")
    log.info("Target documents: %d", config.max_documents)
    log.info("Output directory: %s", config.output_dir)
    
    try:
        # 1. Load Anchor Model
        log.info("\n--- STEP 1: Loading Model ---")
        model, tokenizer = load_anchor_model(config)
        
        # 2. Sample Prompts
        log.info("\n--- STEP 2: Sampling Prompts ---")
        prompts = sample_prompts(config, tokenizer)
        
        if not prompts:
            log.error("No prompts extracted. Aborting.")
            sys.exit(1)
            
        # If we didn't get enough prompts, adjust the target
        if len(prompts) < config.max_documents:
            config.max_documents = len(prompts)
            
        # 3. Generate Dataset
        log.info("\n--- STEP 3: Generating Text ---")
        records, total_time = generate_synthetic_dataset(config, prompts, model, tokenizer)
        
        # 4. Verify Generation
        log.info("\n--- STEP 4: Verifying Data ---")
        if not verify_dataset(config, records):
            log.error("Dataset verification failed! Aborting save.")
            sys.exit(1)
            
        # 5. Save to Arrow
        log.info("\n--- STEP 5: Saving Dataset ---")
        save_to_arrow(config, records)
        
        # 6. Compute Statistics
        log.info("\n--- STEP 6: Generating Statistics ---")
        stats = generate_statistics(config, records, total_time)
        
        # 7. Final config save
        config.save(config.output_dir.parent / f"{config.output_dir.name}_config.json")
        
        log.info("============================================================")
        log.info("  Generation 1 Synthetic Data Pipeline COMPLETED SUCCESSFULLY")
        log.info("============================================================")
        
    except Exception as e:
        log.error("Pipeline encountered a fatal error:")
        log.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
