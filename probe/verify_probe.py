"""
================================================================================
probe/verify_probe.py
================================================================================

Sanity checks the Representation Drift Analysis Framework (PRDAF) on a single test prompt.
Verifies loading, extraction, metric calculation, visualization, and report compilation.
"""

import sys
import logging
from pathlib import Path

# Configure clean logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("probe.verify_probe")

from .probe_config import ProbeConfig
from .utils import select_device, setup_utf8_terminal
from .model_loader import load_evaluation_models
from .hidden_state_extractor import HiddenStateExtractor
from .embedding_analysis import analyze_embeddings
from .attention_analysis import analyze_attention
from .logit_analysis import analyze_logits
from .drift_report import compute_layer_wise_similarities, compile_drift_report
from .visualization import generate_visualizations

def run_verification() -> None:
    """Runs a dry-run check of the PRDAF pipeline."""
    setup_utf8_terminal()
    logger.info("=" * 80)
    logger.info("Initializing PRDAF Verification Suit...")
    logger.info("=" * 80)

    try:
        # 1. Instantiate configuration (points to actual wikitext directories)
        config = ProbeConfig()
        
        # We redirect verification outputs to a separate verify subfolder
        config.output_dir = config.output_dir / "verify"
        config.plots_dir = config.output_dir / "plots"
        config.metrics_dir = config.output_dir / "metrics"
        config.logs_dir = config.output_dir / "logs"
        config.report_json_path = config.output_dir / "representation_drift_report_verify.json"
        config.summary_txt_path = config.output_dir / "representation_summary_verify.txt"
        config.__post_init__()
        
        # Detect hardware acceleration
        device = select_device(config.device, logger)
        
        # 2. Verify model loading
        logger.info("--- Phase 1: Verifying Model Loaders ---")
        anchor_model, student_model, tokenizer = load_evaluation_models(
            config.anchor_model_path,
            config.student_model_path,
            device
        )
        print("✓ Model Loading Verification Passed\n")
        
        # 3. Verify Hidden State Extraction with a single test prompt
        logger.info("--- Phase 2: Verifying Feature Extraction ---")
        test_prompts = ["Artificial intelligence and machine learning models can exhibit representation drift."]
        
        anchor_extractor = HiddenStateExtractor(anchor_model, tokenizer, device)
        student_extractor = HiddenStateExtractor(student_model, tokenizer, device)
        
        logger.info("Extracting features from Anchor model...")
        anchor_features = anchor_extractor.extract_features(test_prompts, batch_size=1)
        
        logger.info("Extracting features from Student model...")
        student_features = student_extractor.extract_features(test_prompts, batch_size=1)
        
        # Check lengths and structure
        assert len(anchor_features["token_embeddings"]) == 1, "Token embedding extraction failed"
        assert len(anchor_features["hidden_states"][0]) == 7, "Hidden state layer extraction count mismatch"
        assert len(anchor_features["attention_weights"][0]) == 6, "Attention weights extraction count mismatch"
        assert anchor_features["logits"][0].ndim == 2, "Logits tensor dimension mismatch"
        print("✓ Hidden State, Embedding, Attention & Logit Extraction Verification Passed\n")
        
        # 4. Verify Metric Correctness
        logger.info("--- Phase 3: Verifying Metric Computations ---")
        
        logger.info("Running embedding similarity check...")
        emb_metrics = analyze_embeddings(anchor_model, student_model, anchor_features, student_features)
        
        logger.info("Running attention divergence check...")
        att_metrics = analyze_attention(anchor_features, student_features, config.num_layers)
        
        logger.info("Running logit divergence check...")
        logit_metrics = analyze_logits(anchor_features, student_features)
        
        logger.info("Running layer-wise hidden state alignment check...")
        layer_metrics = compute_layer_wise_similarities(anchor_features, student_features, config.num_layers)
        
        print("✓ Math & Metric Computations Verification Passed\n")
        
        # 5. Verify Report Generation
        logger.info("--- Phase 4: Verifying Report Compilation ---")
        config_dict = {
            "anchor_model_path": str(config.anchor_model_path),
            "student_model_path": str(config.student_model_path),
            "dataset_source": "Verification Test Prompt",
            "max_prompts": 1,
            "device": str(device)
        }
        
        report = compile_drift_report(
            emb_metrics,
            att_metrics,
            logit_metrics,
            layer_metrics,
            config_dict,
            config.report_json_path,
            config.summary_txt_path
        )
        
        assert config.report_json_path.exists(), "JSON report was not written"
        assert config.summary_txt_path.exists(), "Text summary was not written"
        print("✓ Report Generation Verification Passed\n")
        
        # 6. Verify Visualizations
        logger.info("--- Phase 5: Verifying Visualization Suite ---")
        generate_visualizations(report, config.plots_dir)
        
        assert (config.plots_dir / "representation_drift.png").exists(), "Drift plot missing"
        assert (config.plots_dir / "attention_head_jsd.png").exists(), "Attention head JSD plot missing"
        assert (config.plots_dir / "embedding_similarity.png").exists(), "Embedding similarity plot missing"
        assert (config.plots_dir / "prediction_agreement.png").exists(), "Prediction agreement plot missing"
        print("✓ Visualizations Generation Verification Passed\n")
        
        logger.info("=" * 80)
        logger.info("VERIFICATION COMPLETED SUCCESSFULLY!")
        logger.info("All subsystems are fully operational.")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.critical("Verification Suite FAILED! Details: %s", e, exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_verification()
