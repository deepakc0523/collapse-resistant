"""
================================================================================
ensemble/model_loader.py
================================================================================

Loads the Best Student model for uncertainty evaluation.

Key design decisions:
  - Only the Student model is loaded. The Anchor is NEVER loaded here.
    (Anchor comparison is the Probe module's responsibility.)
  - Model is set to evaluation mode (model.eval()) immediately after loading.
  - All parameters have requires_grad = False — no gradients tracked.
  - No optimizer, no scheduler, no backpropagation.
  - Monte-Carlo Dropout passes are handled by temporarily calling
    model.train() in probability_extractor.py without enabling gradients.
"""

import logging
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger("ensemble.model_loader")


def load_student_model(
    student_path: Path,
    device: torch.device,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Loads the Best Student model from a checkpoint directory and prepares it
    for inference-only uncertainty evaluation.

    Steps performed:
      1. Validate that the checkpoint directory exists.
      2. Load model and tokenizer via HuggingFace AutoClasses.
      3. Set a padding token if one is not defined (required for batching).
      4. Transfer model to the target device.
      5. Switch to evaluation mode.
      6. Freeze all parameters (requires_grad = False).
      7. Log model architecture summary.

    Args:
        student_path: Path to the 'checkpoints/student_model/best' directory.
        device:       Torch device to load the model onto.

    Returns:
        Tuple of (model, tokenizer) — both ready for inference.

    Raises:
        FileNotFoundError: If the student_path directory does not exist.
    """
    if not student_path.exists():
        raise FileNotFoundError(
            f"Student model checkpoint directory not found at: {student_path}\n"
            f"Please ensure the student training phase has completed successfully."
        )

    logger.info("Loading Best Student Model from: %s", student_path)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        str(student_path),
        attn_implementation="eager",  # Avoids flash-attention kernel issues on CPU
    )
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(str(student_path))

    # Ensure padding token exists — required for batched tokenisation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug(
            "No pad_token defined; using eos_token ('%s') as pad_token.",
            tokenizer.eos_token,
        )

    # Transfer to target compute device
    logger.info("Transferring Student model to device: %s", device)
    model = model.to(device)

    # Lock model to evaluation mode — disables BatchNorm running stats update
    model.eval()

    # Freeze all parameters — no gradients tracked under any circumstances
    logger.info("Freezing all parameters (requires_grad = False)...")
    for param in model.parameters():
        param.requires_grad = False

    # Emit a brief architecture summary
    num_params = sum(p.numel() for p in model.parameters())
    num_layers = getattr(model.config, "n_layer", getattr(model.config, "num_hidden_layers", "?"))
    vocab_size = getattr(model.config, "vocab_size", len(tokenizer))

    logger.info(
        "[OK] Student model loaded — Layers: %s | Parameters: %s | Vocab: %s",
        num_layers,
        f"{num_params:,}",
        f"{vocab_size:,}",
    )

    return model, tokenizer
