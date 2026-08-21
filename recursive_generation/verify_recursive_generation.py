"""
================================================================================
recursive_generation/verify_recursive_generation.py
================================================================================

Comprehensive verification suite for the Recursive Generation module.

Validates:
  1. Student model checkpoint loads successfully
  2. GPU is detected (or CPU fallback confirmed)
  3. Prefix dataset loads correctly
  4. Generation produces valid output records
  5. Output JSON schema is correct
  6. Metadata JSON is valid and complete
  7. Resume manager (checkpoint/resume) works correctly
  8. All 4 visualization plots are generated
"""

import sys
import json
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch

from recursive_generation.generation_config import GenerationConfig
from recursive_generation.model_loader import ModelLoader
from recursive_generation.prefix_loader import PrefixLoader
from recursive_generation.generator import SyntheticGenerator
from recursive_generation.resume_manager import ResumeManager
from recursive_generation.metadata_writer import MetadataWriter
from recursive_generation.visualization import GenerationVisualizer
from recursive_generation.utils import get_generation_logger, set_seed, resolve_device


def verify_recursive_generation(limit_prefixes: int = 5) -> bool:
    """
    Runs all verification checks for the Recursive Generation module.

    Parameters
    ----------
    limit_prefixes : int
        Number of prefixes to generate during functional check (keep small).

    Returns
    -------
    bool
        True if all checks pass, False otherwise.
    """
    logger = get_generation_logger("recursive_generation.verify")
    logger.info("=" * 72)
    logger.info(" Starting Recursive Generation Verification Suite...")
    logger.info("=" * 72)

    config = GenerationConfig()
    set_seed(config.random_seed)

    try:
        # --- Check 1: Device Detection ---
        logger.info("[1/8] Detecting compute device...")
        device = resolve_device(config.device)
        logger.info("[PASS] Device: %s | CUDA available: %s", device, torch.cuda.is_available())
        if device.type == "cuda":
            logger.info("  GPU: %s | VRAM: %.1f GB",
                       torch.cuda.get_device_name(0),
                       torch.cuda.get_device_properties(0).total_memory / 1e9)

        # --- Check 2: Student Model Loading ---
        logger.info("[2/8] Loading student model checkpoint...")
        model_loader = ModelLoader(config=config, logger=logger)
        model, tokenizer = model_loader.load(device=device)
        assert model is not None, "Model must load successfully"
        assert tokenizer is not None, "Tokenizer must load successfully"
        logger.info("[PASS] Student model loaded: %s", type(model).__name__)

        # --- Check 3: Prefix Dataset Loading ---
        logger.info("[3/8] Loading prefix dataset...")
        prefix_loader = PrefixLoader(config=config, logger=logger)
        prefixes = prefix_loader.load_prefixes(limit=limit_prefixes)
        assert len(prefixes) > 0, "At least one prefix must be loaded"
        assert all(hasattr(p, "text") for p in prefixes), "All prefixes must have .text attribute"
        logger.info("[PASS] Loaded %d prefixes.", len(prefixes))

        # --- Check 4: Generation Produces Valid Records ---
        logger.info("[4/8] Testing generation pipeline...")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_config = GenerationConfig()
            tmp_config.output_jsonl_path = Path(tmpdir) / "test_output.jsonl"
            tmp_config.resume_checkpoint_path = Path(tmpdir) / "resume_state.json"
            tmp_config.batch_size = min(2, len(prefixes))

            resume_mgr = ResumeManager(config=tmp_config, logger=logger)
            gen = SyntheticGenerator(
                model=model,
                tokenizer=tokenizer,
                config=tmp_config,
                resume_manager=resume_mgr,
                logger=logger,
            )
            stats = gen.generate_all(prefixes, output_path=tmp_config.output_jsonl_path)

            assert stats["successful_generations"] > 0, "At least one record must be generated"
            assert tmp_config.output_jsonl_path.exists(), "Output JSONL must be written"

            # --- Check 5: JSON Schema Validation ---
            logger.info("[5/8] Validating output JSON record schema...")
            with open(tmp_config.output_jsonl_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
            record = json.loads(first_line)
            required_keys = ["prompt", "generated_continuation", "full_text", "generation",
                             "parent_student", "temperature", "top_p", "top_k"]
            for key in required_keys:
                assert key in record, f"Required key '{key}' missing from output record"
            logger.info("[PASS] JSON schema validated: %d required keys present.", len(required_keys))

            # --- Check 6: Resume Manager ---
            logger.info("[6/8] Testing resume checkpoint functionality...")
            resume_mgr.save_state()
            assert tmp_config.resume_checkpoint_path.exists(), "Resume checkpoint must be written"
            state_data = json.loads(tmp_config.resume_checkpoint_path.read_text())
            assert "completed_indices" in state_data, "Resume state must contain completed_indices"

            resume_mgr2 = ResumeManager(config=tmp_config, logger=logger)
            completed = resume_mgr2.load_state()
            assert len(completed) == stats["successful_generations"], \
                f"Resume loaded {len(completed)} but expected {stats['successful_generations']}"
            logger.info("[PASS] Resume checkpoint: %d indices saved and reloaded correctly.", len(completed))

        # --- Check 7: Metadata Writing ---
        logger.info("[7/8] Testing metadata writer...")
        start_time = datetime.now().isoformat()
        writer = MetadataWriter(config=config, logger=logger)
        meta_path = writer.write_metadata(stats, start_time)
        summary_path = writer.write_summary(stats, start_time)
        assert meta_path.exists() and meta_path.stat().st_size > 0
        assert summary_path.exists() and summary_path.stat().st_size > 0
        meta_data = json.loads(meta_path.read_text())
        assert "checkpoint_used" in meta_data, "Metadata must include checkpoint_used"
        assert "sampling_strategy" in meta_data, "Metadata must include sampling_strategy"
        logger.info("[PASS] Metadata and summary written successfully.")

        # --- Check 8: Visualizations ---
        logger.info("[8/8] Testing visualization generation...")
        out_lens = stats.get("output_lengths", [100, 120, 80, 150, 90])
        prm_lens = stats.get("prompt_lengths", [60, 55, 70, 65, 50])
        times = [0.12, 0.14, 0.11, 0.15, 0.13]

        viz = GenerationVisualizer(config=config, logger=logger)
        plots = viz.generate_all_plots(out_lens, prm_lens, times)

        expected_plots = [
            "generation_length_distribution.png",
            "token_length_histogram.png",
            "generation_speed.png",
            "prompt_vs_output_length.png",
        ]
        for pname in expected_plots:
            p = config.plots_dir / pname
            assert p.exists() and p.stat().st_size > 0, f"Missing plot: {pname}"
        logger.info("[PASS] All 4 visualization plots generated successfully.")

        logger.info("=" * 72)
        logger.info(" VERIFICATION COMPLETE: Recursive Generation module is operational!")
        logger.info("=" * 72)
        return True

    except Exception as e:
        logger.error("Verification FAILED: %s", str(e), exc_info=True)
        return False


if __name__ == "__main__":
    success = verify_recursive_generation()
    if not success:
        sys.exit(1)
