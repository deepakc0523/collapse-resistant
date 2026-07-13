"""
================================================================================
probe/model_loader.py
================================================================================

Handles loading and checking the Frozen Anchor and Best Student models.
Ensures both models run in evaluation mode with gradients disabled.
"""

import logging
from pathlib import Path
from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger("probe.model_loader")

def load_evaluation_models(
    anchor_path: Path,
    student_path: Path,
    device: torch.device
) -> Tuple[PreTrainedModel, PreTrainedModel, AutoTokenizer]:
    """
    Loads both Anchor and Student models from specified directories.
    Sets them to evaluation mode, disables gradient tracking, and performs safety validation.
    
    Args:
        anchor_path: Path to the frozen anchor model directory.
        student_path: Path to the best student model directory.
        device: Torch device to load models onto.
        
    Returns:
        Tuple[PreTrainedModel, PreTrainedModel, AutoTokenizer]:
            - Frozen Anchor Model (eval mode, no grad)
            - Best Student Model (eval mode, no grad)
            - Associated Tokenizer
    """
    if not anchor_path.exists():
        raise FileNotFoundError(f"Anchor model checkpoint directory not found at: {anchor_path}")
    if not student_path.exists():
        raise FileNotFoundError(f"Student model checkpoint directory not found at: {student_path}")

    logger.info("Loading Frozen Anchor Model from: %s", anchor_path)
    anchor_model = AutoModelForCausalLM.from_pretrained(str(anchor_path), attn_implementation="eager")
    anchor_tokenizer = AutoTokenizer.from_pretrained(str(anchor_path))

    logger.info("Loading Best Student Model from: %s", student_path)
    student_model = AutoModelForCausalLM.from_pretrained(str(student_path), attn_implementation="eager")
    student_tokenizer = AutoTokenizer.from_pretrained(str(student_path))

    # Standardize padding tokens for generation/batching consistency
    if anchor_tokenizer.pad_token is None:
        anchor_tokenizer.pad_token = anchor_tokenizer.eos_token
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    # 1. Structural and architectural validation
    logger.info("Validating architectural compatibility between Anchor and Student...")
    
    anchor_vocab_size = len(anchor_tokenizer)
    student_vocab_size = len(student_tokenizer)
    if anchor_vocab_size != student_vocab_size:
        raise ValueError(
            f"Vocabulary size mismatch! Anchor tokenizer has {anchor_vocab_size} tokens, "
            f"but Student has {student_vocab_size}."
        )
        
    anchor_layers = anchor_model.config.n_layer
    student_layers = student_model.config.n_layer
    if anchor_layers != student_layers:
        raise ValueError(
            f"Layer count mismatch! Anchor model has {anchor_layers} layers, "
            f"but Student has {student_layers} layers."
        )

    # 2. Transfer to target device
    logger.info("Transferring models to device: %s", device)
    anchor_model = anchor_model.to(device)
    student_model = student_model.to(device)

    # 3. Set to evaluation mode and freeze parameters
    anchor_model.eval()
    student_model.eval()

    logger.info("Disabling gradients (requires_grad = False) for both models...")
    for param in anchor_model.parameters():
        param.requires_grad = False
        
    for param in student_model.parameters():
        param.requires_grad = False

    logger.info("[OK] Both models successfully loaded and locked in evaluation mode.")
    return anchor_model, student_model, anchor_tokenizer
