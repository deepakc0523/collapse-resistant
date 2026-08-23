"""
================================================================================
recursive_generation/generator.py
================================================================================

Core batch generation engine for Generation-2 synthetic data synthesis.

Uses AutoModelForCausalLM.generate() with full hyperparameter support,
batched processing, AMP (float16 on GPU), streaming JSONL writes, tqdm
progress bars, and periodic checkpoint saves for Colab resilience.
"""

import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
from torch.cuda.amp import autocast
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm

from recursive_generation.generation_config import GenerationConfig
from recursive_generation.prefix_loader import PrefixRecord
from recursive_generation.resume_manager import ResumeManager
from recursive_generation.utils import get_generation_logger


class SyntheticGenerator:
    """
    Generates Generation-2 synthetic continuations from human anchor prefixes.

    Processes prompts in configurable batches, writes records as streaming JSONL,
    and saves periodic resume checkpoints for Colab resilience.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: Optional[GenerationConfig] = None,
        resume_manager: Optional[ResumeManager] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        self.resume_manager = resume_manager or ResumeManager(config=self.config)
        self.logger = logger or get_generation_logger("recursive_generation.generator")
        self.device = next(model.parameters()).device

        # Runtime stats
        self._generation_times: List[float] = []
        self._output_lengths: List[int] = []
        self._prompt_lengths: List[int] = []

    def generate_all(
        self,
        prefixes: List[PrefixRecord],
        output_path: Optional[Path] = None,
        available_prefix_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Main generation loop: processes all prefixes in batches, writes output JSONL.

        Parameters
        ----------
        prefixes : List[PrefixRecord]
            Ordered list of prefix records.
        output_path : Optional[Path]
            Override JSONL output path. Uses config path if None.
        available_prefix_count : Optional[int]
            Total available prefixes prior to sampling.

        Returns
        -------
        Dict[str, Any]
            Runtime statistics summary dict.
        """
        out_path = output_path or self.config.output_jsonl_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        avail_count = available_prefix_count if available_prefix_count is not None else len(prefixes)

        # Generator safety guard: never process more than cfg.max_prompts
        if self.config.max_prompts is not None and len(prefixes) > self.config.max_prompts:
            self.logger.warning(
                "Safety guard activated: truncating %d input prefixes to max_prompts (%d).",
                len(prefixes),
                self.config.max_prompts,
            )
            prefixes = prefixes[: self.config.max_prompts]

        selected_count = len(prefixes)

        # Load resume state — skip already-completed indices
        completed_indices = self.resume_manager.load_state()
        pending = [p for p in prefixes if p.index not in completed_indices]

        self.logger.info(
            "Starting Generation-2 synthesis: %d selected prefixes | %d pending | %d already completed",
            len(prefixes),
            len(pending),
            len(completed_indices),
        )

        # Open JSONL in append mode so resumed runs don't overwrite prior records
        open_mode = "a" if completed_indices else "w"

        successful = 0
        failed = 0
        batches_done = 0

        with open(out_path, open_mode, encoding="utf-8") as out_file:
            # Batch the pending prefixes
            batch_size = self.config.batch_size
            batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]

            pbar = tqdm(batches, desc="Generating Generation-2 Synthetic Data", unit="batch")

            for batch in pbar:
                t_start = time.perf_counter()

                try:
                    records = self._generate_batch(batch)
                except Exception as e:
                    self.logger.warning("Batch failed with error: %s", str(e))
                    for pr in batch:
                        self.resume_manager.mark_failed(pr.index)
                        failed += 1
                    continue

                t_end = time.perf_counter()
                batch_time = t_end - t_start

                for rec in records:
                    out_file.write(json.dumps(rec) + "\n")
                    self.resume_manager.mark_completed(rec["_prompt_index"])
                    self._generation_times.append(batch_time / max(len(records), 1))
                    self._output_lengths.append(len(rec["generated_continuation"]))
                    self._prompt_lengths.append(len(rec["prompt"]))
                    successful += 1

                out_file.flush()

                batches_done += 1
                pbar.set_postfix(
                    success=successful,
                    failed=failed,
                    speed=f"{len(batch)/batch_time:.1f} s/batch",
                )

                # Periodic resume checkpoint
                if successful % self.config.checkpoint_every == 0 and successful > 0:
                    self.resume_manager.save_state()
                    self.logger.info("[Checkpoint] Resume state saved at %d successful generations.", successful)

        # Final checkpoint save
        self.resume_manager.save_state()

        total_prefixes = len(prefixes)
        avg_out_len = sum(self._output_lengths) / max(1, len(self._output_lengths))
        avg_prompt_len = sum(self._prompt_lengths) / max(1, len(self._prompt_lengths))
        avg_speed = sum(self._generation_times) / max(1, len(self._generation_times))

        stats = {
            "total_prefixes": total_prefixes,
            "available_prefix_count": avail_count,
            "selected_prefix_count": selected_count,
            "max_prompts": self.config.max_prompts,
            "successful_generations": successful,
            "failed_generations": failed,
            "average_output_char_length": round(avg_out_len, 2),
            "average_prompt_char_length": round(avg_prompt_len, 2),
            "average_time_per_sample_sec": round(avg_speed, 4),
            "output_lengths": self._output_lengths,
            "prompt_lengths": self._prompt_lengths,
        }

        self.logger.info(
            "Generation complete: %d successful | %d failed | avg output len = %.1f chars",
            successful,
            failed,
            avg_out_len,
        )

        return stats

    def _generate_batch(self, batch: List[PrefixRecord]) -> List[Dict[str, Any]]:
        """
        Runs a single batch through model.generate() and decodes outputs.

        Parameters
        ----------
        batch : List[PrefixRecord]
            Batch of prefix records.

        Returns
        -------
        List[Dict[str, Any]]
            Generated record dicts ready for JSONL serialization.
        """
        prompts = [p.text for p in batch]
        cfg = self.config

        # Tokenize with left padding for batch generation
        self.tokenizer.padding_side = "left"
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_prefix_tokens,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        amp_ctx = autocast(dtype=torch.float16) if (self.device.type == "cuda" and cfg.use_amp) else nullcontext()

        with torch.no_grad(), amp_ctx:
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=cfg.max_new_tokens,
                min_new_tokens=cfg.min_new_tokens,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                repetition_penalty=cfg.repetition_penalty,
                num_beams=cfg.num_beams,
                do_sample=cfg.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        records = []
        for i, (pr, out) in enumerate(zip(batch, output_ids)):
            # Decode only the newly generated tokens
            prompt_len = input_ids.shape[1]
            new_tokens = out[prompt_len:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            full_text = pr.text + " " + generated_text

            record = {
                "prompt": pr.text,
                "generated_continuation": generated_text,
                "full_text": full_text,
                "generation": cfg.generation_number,
                "parent_student": cfg.parent_student,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "repetition_penalty": cfg.repetition_penalty,
                "max_new_tokens": cfg.max_new_tokens,
                "seed": cfg.random_seed,
                "_prompt_index": pr.index,
            }
            records.append(record)

        return records
