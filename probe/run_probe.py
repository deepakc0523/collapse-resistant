"""
================================================================================
probe/run_probe.py
================================================================================

Main entry point for the Representation Drift Analysis Framework (PRDAF).
Loads configurations, loads wikitext prompts, extracts representations from
both models, calculates similarity metrics, and outputs reports and plots.
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
logger = logging.getLogger("probe.run_probe")

from .probe_config import ProbeConfig
from .utils import select_device, setup_utf8_terminal, timed_action
from .model_loader import load_evaluation_models
from .prompt_loader import load_prompts
from .hidden_state_extractor import HiddenStateExtractor
from .embedding_analysis import analyze_embeddings
from .attention_analysis import analyze_attention
from .logit_analysis import analyze_logits
from .drift_report import compute_layer_wise_similarities, compile_drift_report
from .visualization import generate_visualizations

@timed_action("Representation Drift Analysis Pipeline", logger)
def main() -> None:
    """Orchestrates the scientific drift measurement suite."""
    setup_utf8_terminal()
    
    logger.info("=" * 80)
    logger.info("STARTING REPRESENTATION DRIFT ANALYSIS FRAMEWORK (PRDAF)")
    logger.info("=" * 80)
    
    # 1. Load Configurations
    config = ProbeConfig()
    
    # 2. Select Hardware Device
    device = select_device(config.device, logger)
    
    # 3. Load Models
    anchor_model, student_model, tokenizer = load_evaluation_models(
        config.anchor_model_path,
        config.student_model_path,
        device
    )
    
    # 4. Load Wikitext Prompts
    prompts = load_prompts(
        config.dataset_source,
        tokenizer,
        max_prompts=config.max_prompts,
        min_tokens=config.prompt_min_tokens,
        max_tokens=config.prompt_max_tokens,
        seed=config.random_seed
    )
    
    # 5. Extract representations for both models
    logger.info("Extracting Frozen Anchor representation states...")
    anchor_extractor = HiddenStateExtractor(anchor_model, tokenizer, device)
    anchor_features = anchor_extractor.extract_features(prompts, batch_size=config.batch_size)
    
    logger.info("Extracting Best Student representation states...")
    student_extractor = HiddenStateExtractor(student_model, tokenizer, device)
    student_features = student_extractor.extract_features(prompts, batch_size=config.batch_size)
    
    # 6. Execute Scientific Analyses
    emb_results = analyze_embeddings(anchor_model, student_model, anchor_features, student_features)
    attn_results = analyze_attention(anchor_features, student_features, config.num_layers)
    logit_results = analyze_logits(anchor_features, student_features)
    layer_results = compute_layer_wise_similarities(anchor_features, student_features, config.num_layers)
    
    # 7. Compile and save JSON/Text reports
    config_dict = {
        "anchor_model_path": str(config.anchor_model_path),
        "student_model_path": str(config.student_model_path),
        "dataset_source": str(config.dataset_source),
        "max_prompts": config.max_prompts,
        "batch_size": config.batch_size,
        "device": str(device)
    }
    
    report = compile_drift_report(
        emb_results,
        attn_results,
        logit_results,
        layer_results,
        config_dict,
        config.report_json_path,
        config.summary_txt_path
    )
    
    # 8. Generate Visualizations
    generate_visualizations(report, config.plots_dir)
    
    logger.info("=" * 80)
    logger.info("REPRESENTATION DRIFT ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("Report JSON saved to : %s", config.report_json_path)
    logger.info("Summary Text saved to : %s", config.summary_txt_path)
    logger.info("Plots generated under: %s", config.plots_dir)
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
