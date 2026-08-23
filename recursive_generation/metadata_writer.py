"""
================================================================================
recursive_generation/metadata_writer.py
================================================================================

Generates and writes generation_metadata.json and generation_summary.txt
for the Generation-2 synthetic dataset.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from recursive_generation.generation_config import GenerationConfig
from recursive_generation.utils import get_generation_logger


class MetadataWriter:
    """Writes generation metadata JSON and plain-text summary files."""

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or GenerationConfig()
        self.logger = logger or get_generation_logger("recursive_generation.metadata_writer")

    def write_metadata(
        self,
        stats: Dict[str, Any],
        run_start_time: str,
    ) -> Path:
        """
        Writes generation_metadata.json artifact.

        Parameters
        ----------
        stats : Dict[str, Any]
            Runtime statistics from SyntheticGenerator.generate_all().
        run_start_time : str
            ISO timestamp of generation run start.

        Returns
        -------
        Path
            Path to written metadata file.
        """
        cfg = self.config
        out_path = cfg.metadata_json_path

        metadata = {
            "metadata": {
                "title": "Generation-2 Synthetic Data Metadata",
                "framework_version": "1.0",
                "generation_timestamp": run_start_time,
                "completion_timestamp": datetime.now().isoformat(),
            },
            "checkpoint_used": str(cfg.student_checkpoint_path),
            "generation_number": cfg.generation_number,
            "parent_student": cfg.parent_student,
            "source_dataset": str(cfg.prefix_dataset_path),
            "max_prompts": cfg.max_prompts,
            "random_seed": cfg.random_seed,
            "available_prefix_count": stats.get("available_prefix_count", stats.get("total_prefixes", 0)),
            "selected_prefix_count": stats.get("selected_prefix_count", stats.get("total_prefixes", 0)),
            "sampling_strategy": {
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "repetition_penalty": cfg.repetition_penalty,
                "max_new_tokens": cfg.max_new_tokens,
                "min_new_tokens": cfg.min_new_tokens,
                "num_beams": cfg.num_beams,
                "do_sample": cfg.do_sample,
                "batch_size": cfg.batch_size,
            },
            "seed": cfg.random_seed,
            "device": cfg.device,
            "use_amp": cfg.use_amp,
            "total_prompts": stats.get("total_prefixes", 0),
            "successful_generations": stats.get("successful_generations", 0),
            "failed_generations": stats.get("failed_generations", 0),
            "average_output_char_length": stats.get("average_output_char_length", 0),
            "average_prompt_char_length": stats.get("average_prompt_char_length", 0),
            "average_time_per_sample_sec": stats.get("average_time_per_sample_sec", 0),
            "output_file": str(cfg.output_jsonl_path),
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        self.logger.info("Wrote generation metadata: %s", out_path)
        return out_path

    def write_summary(
        self,
        stats: Dict[str, Any],
        run_start_time: str,
    ) -> Path:
        """
        Writes human-readable generation_summary.txt artifact.

        Parameters
        ----------
        stats : Dict[str, Any]
            Runtime statistics.
        run_start_time : str
            ISO timestamp of generation run start.

        Returns
        -------
        Path
            Path to written summary file.
        """
        cfg = self.config
        out_path = cfg.summary_txt_path
        total = stats.get("total_prefixes", 0)
        avail = stats.get("available_prefix_count", total)
        selected = stats.get("selected_prefix_count", total)
        success = stats.get("successful_generations", 0)
        failed = stats.get("failed_generations", 0)
        success_rate = success / max(1, total) * 100

        lines = [
            "================================================================================",
            "   RECURSIVE GENERATION MODULE - GENERATION-2 SYNTHESIS REPORT                 ",
            "================================================================================",
            f" Generation Timestamp  : {run_start_time}",
            f" Completion Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f" Checkpoint Used       : {cfg.student_checkpoint_path}",
            f" Source Dataset        : {cfg.prefix_dataset_path}",
            f" Generation Number     : {cfg.generation_number}",
            f" Parent Student        : {cfg.parent_student}",
            f" Max Prompts           : {cfg.max_prompts}",
            f" Available Prefixes    : {avail}",
            f" Selected Prefixes     : {selected}",
            f" Random Seed           : {cfg.random_seed}",
            "--------------------------------------------------------------------------------",
            " GENERATION STATISTICS                                                          ",
            "--------------------------------------------------------------------------------",
            f"  Total Prompts             : {total}",
            f"  Successful Generations    : {success}",
            f"  Failed Generations        : {failed}",
            f"  Success Rate              : {success_rate:.1f}%",
            f"  Avg Output Length         : {stats.get('average_output_char_length', 0):.1f} chars",
            f"  Avg Prompt Length         : {stats.get('average_prompt_char_length', 0):.1f} chars",
            f"  Avg Time / Sample         : {stats.get('average_time_per_sample_sec', 0):.4f} sec",
            "--------------------------------------------------------------------------------",
            " SAMPLING HYPERPARAMETERS                                                       ",
            "--------------------------------------------------------------------------------",
            f"  Temperature               : {cfg.temperature}",
            f"  Top-P (Nucleus Sampling)  : {cfg.top_p}",
            f"  Top-K                     : {cfg.top_k}",
            f"  Repetition Penalty        : {cfg.repetition_penalty}",
            f"  Max New Tokens            : {cfg.max_new_tokens}",
            f"  Min New Tokens            : {cfg.min_new_tokens}",
            f"  Num Beams                 : {cfg.num_beams}",
            f"  Do Sample                 : {cfg.do_sample}",
            f"  Batch Size                : {cfg.batch_size}",
            f"  Random Seed               : {cfg.random_seed}",
            f"  Device                    : {cfg.device}",
            f"  AMP (float16)             : {cfg.use_amp}",
            "================================================================================",
        ]

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        self.logger.info("Wrote generation summary: %s", out_path)
        return out_path
