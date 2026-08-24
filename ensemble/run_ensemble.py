"""
================================================================================
ensemble/run_ensemble.py
================================================================================

Main entry point for the Ensemble Variance Monitor (EVM).

Orchestrates the full EVM pipeline in order:
  1. Configuration and device selection
  2. Student model loading (eval mode, frozen)
  3. Prompt sampling from wikitext dataset
  4. Standard (deterministic) probability extraction
  5. Monte-Carlo Dropout probability extraction
  6. Six uncertainty metric computations
  7. SCRS-ready JSON report + human-readable summary
  8. Five publication-quality visualisations

Usage
-----
    python -m ensemble.run_ensemble
"""

import sys
import logging
from pathlib import Path

# Configure top-level logging before any module imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ensemble.run_ensemble")

from .ensemble_config import EnsembleConfig
from .utils import select_device, setup_utf8_terminal, timed_action, get_cuda_memory_report
from .model_loader import load_student_model
from .prompt_loader import load_prompts
from .probability_extractor import ProbabilityExtractor
from .uncertainty_metrics import compute_all_metrics
from .variance_report import compile_variance_report
from .visualization import generate_visualizations


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble Variance Monitor (EVM)")
    parser.add_argument("--student-model-path", type=Path, default=None, help="Path to student model checkpoint")
    parser.add_argument("--dataset-source", type=Path, default=None, help="Path to prompt source dataset")
    parser.add_argument("--output-dir", type=Path, default=None, help="Path to output directory for ensemble artifacts")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment identifier name")
    return parser.parse_args()

@timed_action("Ensemble Variance Monitor Pipeline", logger)
def main() -> None:
    """Runs the complete EVM uncertainty measurement pipeline."""

    setup_utf8_terminal()
    args = parse_args()

    logger.info("=" * 80)
    logger.info("STARTING ENSEMBLE VARIANCE MONITOR (EVM)")
    logger.info("Measuring Student Model Prediction Uncertainty")
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Step 1: Configuration
    # ------------------------------------------------------------------
    logger.info("[Step 1/8] Loading EVM configuration...")
    config = EnsembleConfig()
    config.update_paths(
        output_dir=args.output_dir,
        student_model_path=args.student_model_path,
        dataset_source=args.dataset_source,
    )

    logger.info("  Student model : %s", config.student_model_path)
    logger.info("  Dataset       : %s", config.dataset_source)
    logger.info("  Output dir    : %s", config.output_dir)
    logger.info("  Max prompts   : %d", config.max_prompts)
    logger.info("  Batch size    : %d", config.batch_size)
    logger.info("  MC passes     : %d", config.mc_dropout_passes)
    logger.info("  Top-k         : %d", config.top_k_confidence)
    logger.info("  Random seed   : %d", config.random_seed)

    # ------------------------------------------------------------------
    # Step 2: Hardware device
    # ------------------------------------------------------------------
    logger.info("[Step 2/8] Selecting compute device...")
    device = select_device(config.device, logger)

    # ------------------------------------------------------------------
    # Step 3: Load Student model
    # ------------------------------------------------------------------
    logger.info("[Step 3/8] Loading Best Student model...")
    model, tokenizer = load_student_model(config.student_model_path, device)
    vocab_size: int = getattr(model.config, "vocab_size", len(tokenizer))

    # ------------------------------------------------------------------
    # Step 4: Load prompts
    # ------------------------------------------------------------------
    logger.info("[Step 4/8] Loading prompts from dataset...")
    prompts = load_prompts(
        config.dataset_source,
        tokenizer,
        max_prompts=config.max_prompts,
        min_tokens=config.prompt_min_tokens,
        max_tokens=config.prompt_max_tokens,
        seed=config.random_seed,
    )
    logger.info("  Loaded %d prompts (seed=%d).", len(prompts), config.random_seed)

    # ------------------------------------------------------------------
    # Step 5: Standard probability extraction
    # ------------------------------------------------------------------
    logger.info("[Step 5/8] Running deterministic forward passes...")
    extractor = ProbabilityExtractor(model, tokenizer, device)
    standard_extraction = extractor.extract_probabilities(prompts, batch_size=config.batch_size)

    if device.type == "cuda":
        logger.info("CUDA memory after standard extraction: %s", get_cuda_memory_report())

    # ------------------------------------------------------------------
    # Step 6: Monte-Carlo Dropout extraction
    # ------------------------------------------------------------------
    logger.info(
        "[Step 6/8] Running Monte-Carlo Dropout (%d passes per prompt)...",
        config.mc_dropout_passes,
    )
    mc_extraction = extractor.extract_mc_dropout_probabilities(
        prompts,
        n_passes=config.mc_dropout_passes,
        batch_size=config.batch_size,
    )

    if device.type == "cuda":
        logger.info("CUDA memory after MC Dropout: %s", get_cuda_memory_report())

    # ------------------------------------------------------------------
    # Step 7: Compute all six uncertainty metrics
    # ------------------------------------------------------------------
    logger.info("[Step 7/8] Computing uncertainty metrics...")
    metrics_result = compute_all_metrics(
        softmax_probs=standard_extraction["softmax_probs"],
        mc_predictions=mc_extraction["mc_predictions"],
        top_k=config.top_k_confidence,
        vocab_size=vocab_size,
    )

    # Log aggregate summary
    agg = metrics_result["aggregate"]
    logger.info("  --- Aggregate Uncertainty Metrics ---")
    logger.info("  Predictive Entropy    : %.6f  (std: %.6f)", agg["mean_predictive_entropy"],    agg["std_predictive_entropy"])
    logger.info("  Top-1 Confidence      : %.6f  (std: %.6f)", agg["mean_top1_confidence"],       agg["std_top1_confidence"])
    logger.info("  Top-5 Spread          : %.6f  (std: %.6f)", agg["mean_top5_confidence_spread"],agg["std_top5_confidence_spread"])
    logger.info("  Probability Variance  : %.6f  (std: %.6f)", agg["mean_probability_variance"],  agg["std_probability_variance"])
    logger.info("  Confidence Margin     : %.6f  (std: %.6f)", agg["mean_confidence_margin"],     agg["std_confidence_margin"])
    logger.info("  MC Dropout Consistency: %.6f  (std: %.6f)", agg["mean_mc_dropout_consistency"],agg["std_mc_dropout_consistency"])

    # ------------------------------------------------------------------
    # Step 8a: Compile and save reports
    # ------------------------------------------------------------------
    logger.info("[Step 8/8a] Compiling reports...")
    config_snapshot = {
        "student_model_path": str(config.student_model_path),
        "dataset_source":     str(config.dataset_source),
        "max_prompts":        config.max_prompts,
        "prompts_processed":  len(prompts),
        "batch_size":         config.batch_size,
        "mc_dropout_passes":  config.mc_dropout_passes,
        "top_k_confidence":   config.top_k_confidence,
        "device":             str(device),
        "random_seed":        config.random_seed,
        "vocab_size":         vocab_size,
    }

    report = compile_variance_report(
        metrics_result,
        config_snapshot,
        config.report_json_path,
        config.summary_txt_path,
    )

    # ------------------------------------------------------------------
    # Step 8b: Generate visualizations
    # ------------------------------------------------------------------
    logger.info("[Step 8/8b] Generating visualisations...")
    generate_visualizations(report, config.plots_dir)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("ENSEMBLE VARIANCE MONITOR PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("")
    logger.info("Output artifacts:")
    logger.info("  JSON report  : %s", config.report_json_path)
    logger.info("  Text summary : %s", config.summary_txt_path)
    logger.info("  Plots dir    : %s", config.plots_dir)
    logger.info("")
    logger.info("Plots generated:")
    logger.info("  entropy_distribution.png")
    logger.info("  confidence_histogram.png")
    logger.info("  variance_histogram.png")
    logger.info("  confidence_margin.png")
    logger.info("  mc_dropout_consistency.png")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
