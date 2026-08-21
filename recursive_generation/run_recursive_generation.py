"""
================================================================================
recursive_generation/run_recursive_generation.py
================================================================================

Main pipeline runner for Generation-2 synthetic data production.

Pipeline:
  Student Checkpoint + Prefix Dataset
      -> ModelLoader
      -> PrefixLoader
      -> ResumeManager (loads prior state if interrupted)
      -> SyntheticGenerator (batched GPU generation with AMP + tqdm)
      -> MetadataWriter
      -> GenerationVisualizer
      -> recursive_generation_out/generation_2/
"""

import sys
import logging
from datetime import datetime

from recursive_generation.generation_config import GenerationConfig
from recursive_generation.model_loader import ModelLoader
from recursive_generation.prefix_loader import PrefixLoader
from recursive_generation.generator import SyntheticGenerator
from recursive_generation.resume_manager import ResumeManager
from recursive_generation.metadata_writer import MetadataWriter
from recursive_generation.visualization import GenerationVisualizer
from recursive_generation.utils import get_generation_logger, set_seed, resolve_device


def run_recursive_generation() -> None:
    """Executes the full Generation-2 synthesis pipeline."""

    logger = get_generation_logger("recursive_generation.run")
    logger.info("=" * 72)
    logger.info(" Recursive Generation Module - Generation-2 Synthesis")
    logger.info("=" * 72)

    run_start_time = datetime.now().isoformat()

    config = GenerationConfig()
    set_seed(config.random_seed)

    # 1. Resolve device
    device = resolve_device(config.device)
    logger.info("Compute device: %s", device)

    # 2. Load student model
    model_loader = ModelLoader(config=config, logger=logger)
    model, tokenizer = model_loader.load(device=device)

    # 3. Load human anchor prefixes
    prefix_loader = PrefixLoader(config=config, logger=logger)
    prefixes = prefix_loader.load_prefixes()

    # 4. Initialize resume manager (enables Colab restart recovery)
    resume_mgr = ResumeManager(config=config, logger=logger)

    # 5. Run batch generation with streaming JSONL writes
    gen = SyntheticGenerator(
        model=model,
        tokenizer=tokenizer,
        config=config,
        resume_manager=resume_mgr,
        logger=logger,
    )
    stats = gen.generate_all(prefixes)

    # 6. Write metadata and summary
    writer = MetadataWriter(config=config, logger=logger)
    meta_path = writer.write_metadata(stats, run_start_time)
    summary_path = writer.write_summary(stats, run_start_time)

    # 7. Generate visualizations
    viz = GenerationVisualizer(config=config, logger=logger)
    plots = viz.generate_all_plots(
        output_lengths=stats.get("output_lengths", []),
        prompt_lengths=stats.get("prompt_lengths", []),
        generation_times=[stats.get("average_time_per_sample_sec", 0.1)],
    )

    logger.info("-" * 72)
    logger.info(" Generation-2 Synthesis Complete:")
    logger.info("  Successful : %d", stats.get("successful_generations", 0))
    logger.info("  Failed     : %d", stats.get("failed_generations", 0))
    logger.info("  JSONL      : %s", config.output_jsonl_path)
    logger.info("  Metadata   : %s", meta_path)
    logger.info("  Summary    : %s", summary_path)
    logger.info("  Plots      : %d files in %s", len(plots), config.plots_dir)
    logger.info("=" * 72)


if __name__ == "__main__":
    run_recursive_generation()
