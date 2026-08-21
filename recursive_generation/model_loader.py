"""
================================================================================
recursive_generation/model_loader.py
================================================================================

Loads the Generation-1 student model checkpoint for Generation-2 synthesis.

Uses AutoModelForCausalLM so the loader is model-agnostic and compatible
with any Hugging Face causal LM checkpoint (distilgpt2, gpt2, etc.).
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from recursive_generation.generation_config import GenerationConfig
from recursive_generation.utils import get_generation_logger


class ModelLoader:
    """Loads and prepares the trained student model for generation."""

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or GenerationConfig()
        self.logger = logger or get_generation_logger("recursive_generation.model_loader")

    def load(
        self,
        checkpoint_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """
        Loads the student causal LM and tokenizer from a Hugging Face checkpoint directory.

        Parameters
        ----------
        checkpoint_path : Optional[Path]
            Override path to the checkpoint directory. Uses config path if None.
        device : Optional[torch.device]
            Target device. Uses config device if None.

        Returns
        -------
        Tuple[PreTrainedModel, PreTrainedTokenizerBase]
            Loaded model and tokenizer.
        """
        ckpt = checkpoint_path or self.config.student_checkpoint_path
        dev = device or torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        self.logger.info("Loading student checkpoint from: %s", ckpt)
        self.logger.info("Target device: %s", dev)

        if not ckpt.exists():
            raise FileNotFoundError(
                f"Student checkpoint not found: {ckpt}. "
                "Please ensure student training has been completed and checkpointed."
            )

        # Load tokenizer from checkpoint directory
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load causal LM model
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt),
            torch_dtype=torch.float16 if (dev.type == "cuda" and self.config.use_amp) else torch.float32,
        )
        model = model.to(dev)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        self.logger.info(
            "Student model loaded successfully: %.2f M parameters | dtype=%s",
            n_params / 1_000_000,
            next(model.parameters()).dtype,
        )

        return model, tokenizer
