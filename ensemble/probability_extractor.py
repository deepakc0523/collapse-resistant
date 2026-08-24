"""
================================================================================
ensemble/probability_extractor.py
================================================================================

Extracts softmax probability distributions from the Student model.

Two extraction modes:

1. Standard (deterministic):
   model.eval() + torch.no_grad()
   Single forward pass per batch → softmax(logits)

2. Monte-Carlo (MC) Dropout:
   model.train() (activates dropout layers) + torch.no_grad()
   N stochastic forward passes per prompt → N probability distributions
   All parameters remain frozen (requires_grad = False).

All tensors are moved to CPU immediately after each batch to prevent
accumulation of GPU memory during long evaluation runs.
"""

import logging
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger("ensemble.probability_extractor")


class ProbabilityExtractor:
    """
    Orchestrates forward passes through the Student model and extracts
    softmax probability distributions over the vocabulary.

    Attributes
    ----------
    model : PreTrainedModel
        The frozen Best Student model.
    tokenizer : PreTrainedTokenizer
        Associated tokenizer (with pad_token set).
    device : torch.device
        Compute device.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_probabilities(
        self,
        prompts: List[str],
        batch_size: int = 2,
    ) -> Dict[str, Any]:
        """
        Runs a single deterministic forward pass over all prompts and
        extracts softmax probability distributions.

        Args:
            prompts:    List of prompt strings.
            batch_size: Number of prompts per forward pass.

        Returns:
            Dict with keys:
              - "softmax_probs"     : List[Tensor[seq_len, vocab_size]] — per token
              - "logits"            : List[Tensor[seq_len, vocab_size]] — raw logits
              - "predicted_tokens"  : List[Tensor[seq_len]]             — argmax tokens
        """
        self.model.eval()

        result: Dict[str, Any] = {
            "softmax_probs": [],
            "logits": [],
            "predicted_tokens": [],
        }

        total_prompts = len(prompts)
        logger.info(
            "Extracting standard (deterministic) probabilities for %d prompts "
            "(batch_size=%d)...",
            total_prompts,
            batch_size,
        )

        with torch.no_grad():
            for batch_start in range(0, total_prompts, batch_size):
                batch_prompts = prompts[batch_start : batch_start + batch_size]
                batch_probs, batch_logits, batch_preds = self._forward_batch(batch_prompts)

                result["softmax_probs"].extend(batch_probs)
                result["logits"].extend(batch_logits)
                result["predicted_tokens"].extend(batch_preds)

                logger.debug(
                    "Processed prompts %d–%d / %d",
                    batch_start + 1,
                    min(batch_start + batch_size, total_prompts),
                    total_prompts,
                )

        logger.info(
            "[OK] Standard probability extraction complete. %d prompt distributions ready.",
            len(result["softmax_probs"]),
        )
        return result

    def extract_mc_dropout_probabilities(
        self,
        prompts: List[str],
        n_passes: int = 10,
        batch_size: int = 2,
    ) -> Dict[str, Any]:
        """
        Runs N stochastic forward passes with dropout activated to extract top-1
        predicted token IDs per pass.

        Monte-Carlo Dropout activates dropout layers by calling model.train(),
        while torch.no_grad() ensures no gradient computation or storage.
        Parameters remain fully frozen (requires_grad = False throughout).

        Only top-1 predicted token IDs are retained on CPU to prevent excessive
        CPU/GPU memory usage during long evaluation runs.

        Args:
            prompts:    List of prompt strings.
            n_passes:   Number of stochastic forward passes.
            batch_size: Number of prompts per forward pass.

        Returns:
            Dict with keys:
              - "mc_predictions": List[List[Tensor[seq_len]]]
                  Outer list indexed by prompt.
                  Inner list indexed by MC pass (length = n_passes).
        """
        logger.info(
            "Starting Monte-Carlo Dropout extraction: %d passes × %d prompts...",
            n_passes,
            len(prompts),
        )

        # Initialise per-prompt storage: mc_predictions[prompt_idx][pass_idx]
        num_prompts = len(prompts)
        mc_predictions: List[List[torch.Tensor]] = [[] for _ in range(num_prompts)]

        # Temporarily switch to train mode to activate dropout stochasticity
        self.model.train()

        try:
            with torch.no_grad():
                for pass_idx in range(n_passes):
                    logger.info("MC Dropout pass %d/%d", pass_idx + 1, n_passes)

                    for batch_start in range(0, num_prompts, batch_size):
                        batch_prompts = prompts[batch_start : batch_start + batch_size]
                        
                        inputs = self.tokenizer(
                            batch_prompts,
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                        )
                        input_ids = inputs["input_ids"].to(self.device)
                        attention_mask = inputs["attention_mask"].to(self.device)

                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True,
                        )

                        logits_batch = outputs.logits
                        preds_batch = torch.argmax(logits_batch, dim=-1)

                        preds_cpu = preds_batch.cpu()
                        mask_cpu = attention_mask.cpu()

                        # Free temporary GPU tensors immediately
                        del outputs, logits_batch, preds_batch, input_ids, attention_mask
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()

                        b_size = preds_cpu.size(0)
                        for b in range(b_size):
                            actual_len = int(mask_cpu[b].sum().item())
                            global_idx = batch_start + b
                            mc_predictions[global_idx].append(preds_cpu[b, :actual_len])

        finally:
            # Always restore evaluation mode after MC passes
            self.model.eval()

        logger.info(
            "[OK] MC Dropout extraction complete. %d passes per prompt.", n_passes
        )
        return {"mc_predictions": mc_predictions}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _forward_batch(
        self,
        batch_prompts: List[str],
    ) -> tuple:
        """
        Tokenises and runs a single batch of prompts through the model.

        Returns:
            Tuple of:
              - List[Tensor[seq_len, vocab_size]] — softmax probs (CPU, one per prompt)
              - List[Tensor[seq_len, vocab_size]] — raw logits    (CPU, one per prompt)
              - List[Tensor[seq_len]]             — argmax tokens (CPU, one per prompt)
        """
        inputs = self.tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # outputs.logits shape: [batch, seq_len, vocab_size]
        logits_batch = outputs.logits  # on device
        probs_batch = F.softmax(logits_batch, dim=-1)  # on device
        preds_batch = torch.argmax(logits_batch, dim=-1)  # on device

        # Move to CPU and disassemble into per-prompt tensors
        logits_cpu = logits_batch.cpu()
        probs_cpu = probs_batch.cpu()
        preds_cpu = preds_batch.cpu()
        mask_cpu = attention_mask.cpu()

        batch_probs: List[torch.Tensor] = []
        batch_logits: List[torch.Tensor] = []
        batch_preds: List[torch.Tensor] = []

        batch_size = input_ids.size(0)
        for b in range(batch_size):
            actual_len = int(mask_cpu[b].sum().item())
            batch_probs.append(probs_cpu[b, :actual_len])    # [actual_len, vocab]
            batch_logits.append(logits_cpu[b, :actual_len])  # [actual_len, vocab]
            batch_preds.append(preds_cpu[b, :actual_len])    # [actual_len]

        return batch_probs, batch_logits, batch_preds
