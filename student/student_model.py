"""
================================================================================
student/student_model.py
================================================================================

Student Model Architecture.
Initializes a DistilGPT2 model with completely RANDOM WEIGHTS.
This is a critical constraint for the recursive synthetic learning experiment.
No pretrained weights or anchor weights are used.
"""

from typing import Tuple, Optional
import torch
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

def get_tokenizer(model_type: str = "distilgpt2") -> PreTrainedTokenizer:
    """
    Load the GPT2 tokenizer. 
    The vocabulary must match the Anchor model exactly.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_random_student_model(
    model_type: str = "distilgpt2",
    device: Optional[torch.device] = None,
    logger = None
) -> PreTrainedModel:
    """
    Initializes a new DistilGPT2 model from scratch.
    
    CRITICAL: We use GPT2Config and GPT2LMHeadModel directly without calling
    from_pretrained() on the model weights, ensuring random initialization.
    """
    if logger:
        logger.info("Initializing Student Model with RANDOM WEIGHTS: %s", model_type)
        
    # 1. Load the configuration for the architecture (not the weights)
    config = GPT2Config.from_pretrained(model_type)
    
    # 2. Initialize the model with random weights based on the config
    model = GPT2LMHeadModel(config)
    
    if device is not None:
        model = model.to(device)
        
    n_params = sum(p.numel() for p in model.parameters())
    if logger:
        logger.info("Student Model initialized: %d M parameters", n_params // 1_000_000)
        
    return model
