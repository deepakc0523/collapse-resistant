"""
================================================================================
ensemble/verify_ensemble.py
================================================================================

Sanity-checker for the Ensemble Variance Monitor (EVM).

Runs the pipeline on a single hard-coded test prompt and verifies all
subsystems phase-by-phase. The verification uses a separate output
subdirectory (ensemble_out/verify/) to avoid polluting production output.

Verification phases
-------------------
Phase 1 — Student model loads (eval mode, frozen params)
Phase 2 — Prompts load from dataset (falls back to test prompt on failure)
Phase 3 — Softmax probabilities extracted from Student
Phase 4 — Entropy computed and within [0, 1]
Phase 5 — Variance computed and within [0, 1]
Phase 6 — Top-1 Confidence, Margin, and Spread within [0, 1]
Phase 7 — MC Dropout runs and consistency within [0, 1]
Phase 8 — JSON and TXT reports generated
Phase 9 — All 5 plots generated

Usage
-----
    python -m ensemble.verify_ensemble
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ensemble.verify_ensemble")

from .ensemble_config import EnsembleConfig
from .utils import select_device, setup_utf8_terminal
from .model_loader import load_student_model
from .probability_extractor import ProbabilityExtractor
from .uncertainty_metrics import (
    compute_predictive_entropy,
    compute_top1_confidence,
    compute_top5_confidence_spread,
    compute_probability_variance,
    compute_confidence_margin,
    compute_mc_dropout_consistency,
    compute_all_metrics,
)
from .variance_report import compile_variance_report
from .visualization import generate_visualizations


# Single well-formed test prompt used when the dataset is unavailable
_VERIFICATION_PROMPT = (
    "Language models trained on synthetic data may exhibit uncertainty in "
    "their probability distributions, particularly in later generation cycles."
)

_TOLERANCE = 1e-9  # Floating-point guard for [0, 1] bound checks


def _check_range(value: float, name: str, low: float = 0.0, high: float = 1.0) -> None:
    """Asserts that a scalar value lies within [low, high]."""
    import math
    if math.isnan(value):
        raise AssertionError(f"{name} is NaN — computation failed.")
    if not (low - _TOLERANCE <= value <= high + _TOLERANCE):
        raise AssertionError(
            f"{name} = {value:.8f} is outside expected range [{low}, {high}]."
        )


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Sanity-checker for Ensemble Variance Monitor (EVM)")
    parser.add_argument("--student-model-path", type=Path, default=None, help="Path to student model checkpoint")
    parser.add_argument("--dataset-source", type=Path, default=None, help="Path to prompt source dataset")
    parser.add_argument("--output-dir", type=Path, default=None, help="Path to output directory for verification artifacts")
    return parser.parse_args()


def run_verification() -> None:
    """Executes all nine verification phases against the EVM pipeline."""
    setup_utf8_terminal()
    args = parse_args()

    logger.info("=" * 80)
    logger.info("ENSEMBLE VARIANCE MONITOR (EVM) — VERIFICATION SUITE")
    logger.info("=" * 80)

    config = EnsembleConfig()
    config.update_paths(
        output_dir=args.output_dir,
        student_model_path=args.student_model_path,
        dataset_source=args.dataset_source,
    )

    # Redirect all verification outputs to a dedicated subdirectory
    config.output_dir      = config.output_dir / "verify"
    config.plots_dir       = config.output_dir / "plots"
    config.report_json_path = config.output_dir / "variance_report_verify.json"
    config.summary_txt_path = config.output_dir / "variance_summary_verify.txt"
    config.__post_init__()

    # Reduce prompts for speed in verification
    config.max_prompts     = 1
    config.batch_size      = 1
    config.mc_dropout_passes = 5

    logger.info("Verifying Student model at: %s", config.student_model_path)

    try:
        # ------------------------------------------------------------------
        # Phase 1: Student model loading
        # ------------------------------------------------------------------
        logger.info("--- Phase 1: Verifying Student Model Loading ---")
        device = select_device(config.device, logger)
        model, tokenizer = load_student_model(config.student_model_path, device)

        # Check that the model is in eval mode
        assert not model.training, "Model is in training mode — eval() not applied."

        # Check that no parameters are trainable
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == 0, (
            f"{trainable:,} parameters still have requires_grad=True — freeze failed."
        )

        vocab_size: int = getattr(model.config, "vocab_size", len(tokenizer))
        print("✓ Phase 1 PASSED — Student model loaded, eval mode, all params frozen.\n")

        # ------------------------------------------------------------------
        # Phase 2: Prompt loading
        # ------------------------------------------------------------------
        logger.info("--- Phase 2: Verifying Prompt Loading ---")
        try:
            from .prompt_loader import load_prompts
            prompts = load_prompts(
                config.dataset_source,
                tokenizer,
                max_prompts=1,
                min_tokens=config.prompt_min_tokens,
                max_tokens=config.prompt_max_tokens,
                seed=config.random_seed,
            )
            assert len(prompts) >= 1, "Prompt loader returned an empty list."
            # Use only the first prompt for speed
            prompts = prompts[:1]
            logger.info("Loaded prompt from dataset: '%s...'", prompts[0][:60])
        except FileNotFoundError:
            logger.warning(
                "Dataset not found at %s — using built-in verification prompt.",
                config.dataset_source,
            )
            prompts = [_VERIFICATION_PROMPT]

        assert len(prompts) >= 1
        print("✓ Phase 2 PASSED — Prompts loaded successfully.\n")

        # ------------------------------------------------------------------
        # Phase 3: Probability extraction
        # ------------------------------------------------------------------
        logger.info("--- Phase 3: Verifying Probability Extraction ---")
        extractor = ProbabilityExtractor(model, tokenizer, device)
        extraction = extractor.extract_probabilities(prompts, batch_size=1)

        probs_list = extraction["softmax_probs"]
        assert len(probs_list) == len(prompts), "Mismatch in number of extracted probability tensors."

        first_probs = probs_list[0]  # [seq_len, vocab_size]
        assert first_probs.ndim == 2, f"Expected 2D tensor, got {first_probs.ndim}D."
        assert first_probs.size(-1) == vocab_size, (
            f"Vocabulary dimension mismatch: expected {vocab_size}, got {first_probs.size(-1)}."
        )

        # Verify probabilities sum to ~1 per token position
        row_sums = first_probs.sum(dim=-1)
        max_deviation = (row_sums - 1.0).abs().max().item()
        assert max_deviation < 1e-4, (
            f"Softmax rows do not sum to 1 (max deviation: {max_deviation:.2e})."
        )

        print("✓ Phase 3 PASSED — Softmax probabilities extracted and validated.\n")

        # ------------------------------------------------------------------
        # Phase 4: Entropy metric
        # ------------------------------------------------------------------
        logger.info("--- Phase 4: Verifying Predictive Entropy ---")
        entropy = compute_predictive_entropy(first_probs, vocab_size)
        _check_range(entropy, "Predictive Entropy")
        logger.info("  Predictive Entropy = %.6f", entropy)
        print("✓ Phase 4 PASSED — Entropy in [0, 1].\n")

        # ------------------------------------------------------------------
        # Phase 5: Variance metric
        # ------------------------------------------------------------------
        logger.info("--- Phase 5: Verifying Probability Variance ---")
        variance = compute_probability_variance(first_probs, top_k=config.top_k_confidence)
        _check_range(variance, "Probability Variance")
        logger.info("  Probability Variance = %.6f", variance)
        print("✓ Phase 5 PASSED — Variance in [0, 1].\n")

        # ------------------------------------------------------------------
        # Phase 6: Confidence metrics (Top-1, Margin, Spread)
        # ------------------------------------------------------------------
        logger.info("--- Phase 6: Verifying Confidence Metrics ---")
        top1_conf = compute_top1_confidence(first_probs)
        margin    = compute_confidence_margin(first_probs)
        spread    = compute_top5_confidence_spread(first_probs, top_k=config.top_k_confidence)

        _check_range(top1_conf, "Top-1 Confidence")
        _check_range(margin,    "Confidence Margin")
        _check_range(spread,    "Top-5 Confidence Spread")

        logger.info("  Top-1 Confidence = %.6f", top1_conf)
        logger.info("  Confidence Margin = %.6f", margin)
        logger.info("  Top-5 Spread = %.6f", spread)
        print("✓ Phase 6 PASSED — Top-1, Margin, and Spread all in [0, 1].\n")

        # ------------------------------------------------------------------
        # Phase 7: Monte-Carlo Dropout
        # ------------------------------------------------------------------
        logger.info("--- Phase 7: Verifying Monte-Carlo Dropout ---")
        mc_extraction = extractor.extract_mc_dropout_probabilities(
            prompts, n_passes=config.mc_dropout_passes, batch_size=1
        )
        mc_predictions = mc_extraction["mc_predictions"]

        assert len(mc_predictions) == len(prompts), "MC Dropout: prompt count mismatch."
        assert len(mc_predictions[0]) == config.mc_dropout_passes, (
            f"MC Dropout: expected {config.mc_dropout_passes} passes, "
            f"got {len(mc_predictions[0])}."
        )

        first_pred = mc_predictions[0][0]
        assert first_pred.ndim == 1, f"Expected 1D prediction tensor, got {first_pred.ndim}D"
        assert first_pred.device.type == "cpu", f"Expected CPU tensor, got {first_pred.device}"
        assert first_pred.size(0) == first_probs.size(0), (
            f"Sequence length mismatch: expected {first_probs.size(0)}, got {first_pred.size(0)}"
        )

        consistency = compute_mc_dropout_consistency(mc_predictions[0])
        _check_range(consistency, "MC Dropout Consistency")
        logger.info("  MC Dropout Consistency = %.6f  (over %d passes)", consistency, config.mc_dropout_passes)
        print("✓ Phase 7 PASSED — MC Dropout runs and consistency in [0, 1].\n")

        # ------------------------------------------------------------------
        # Phase 8: Report generation
        # ------------------------------------------------------------------
        logger.info("--- Phase 8: Verifying Report Generation ---")
        all_metrics = compute_all_metrics(
            softmax_probs=probs_list,
            mc_predictions=mc_predictions,
            top_k=config.top_k_confidence,
            vocab_size=vocab_size,
        )

        config_snapshot = {
            "student_model_path": str(config.student_model_path),
            "dataset_source":     "Verification Test Prompt",
            "max_prompts":        1,
            "batch_size":         config.batch_size,
            "mc_dropout_passes":  config.mc_dropout_passes,
            "top_k_confidence":   config.top_k_confidence,
            "device":             str(device),
            "random_seed":        config.random_seed,
        }

        report = compile_variance_report(
            all_metrics,
            config_snapshot,
            config.report_json_path,
            config.summary_txt_path,
        )

        assert config.report_json_path.exists(), "JSON report was not written to disk."
        assert config.summary_txt_path.exists(), "Text summary was not written to disk."
        print("✓ Phase 8 PASSED — JSON and TXT reports generated.\n")

        # ------------------------------------------------------------------
        # Phase 9: Visualizations
        # ------------------------------------------------------------------
        logger.info("--- Phase 9: Verifying Visualizations ---")
        generate_visualizations(report, config.plots_dir)

        expected_plots = [
            "entropy_distribution.png",
            "confidence_histogram.png",
            "variance_histogram.png",
            "confidence_margin.png",
            "mc_dropout_consistency.png",
        ]
        for plot_name in expected_plots:
            plot_path = config.plots_dir / plot_name
            assert plot_path.exists(), f"Expected plot not found: {plot_path}"

        print("✓ Phase 9 PASSED — All 5 plots generated.\n")

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        logger.info("=" * 80)
        logger.info("ALL VERIFICATION PHASES PASSED SUCCESSFULLY!")
        logger.info("Verification outputs written to: %s", config.output_dir)
        logger.info("=" * 80)

    except AssertionError as ae:
        logger.critical("VERIFICATION ASSERTION FAILED: %s", ae, exc_info=False)
        sys.exit(1)
    except Exception as exc:
        logger.critical("VERIFICATION SUITE FAILED with unexpected error.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_verification()
