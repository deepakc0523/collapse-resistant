"""
================================================================================
generation/generator.py
================================================================================

Project: Anchor-Regularized Model Collapse Prevention
Component: Synthetic Data Generation Pipeline (Generation 1)

Purpose:
    Core generation engine. Loads the Frozen Anchor model and executes batched
    text generation using Top-p/Top-k sampling. 

Authors: Deepak (Research Lead)
Created: 2026-07-09
License: MIT
================================================================================
"""

import time
import uuid
import datetime
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from .generation_config import GenerationConfig
from .utils import setup_logger

log = setup_logger("generator")


def load_anchor_model(config: GenerationConfig):
    """Load the frozen Anchor model and tokenizer."""
    log.info("Loading Frozen Anchor from: %s", config.frozen_anchor_dir)
    
    if not config.frozen_anchor_dir.exists():
        raise FileNotFoundError(f"Frozen Anchor not found at {config.frozen_anchor_dir}")
        
    tokenizer = AutoTokenizer.from_pretrained(str(config.frozen_anchor_dir))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(str(config.frozen_anchor_dir))
    model.to(config.device)
    model.eval()  # strictly evaluation mode
    
    # Sanity check freezing
    for param in model.parameters():
        param.requires_grad = False
        
    return model, tokenizer


def generate_synthetic_dataset(
    config: GenerationConfig, 
    prompts: List[Tuple[str, str]],
    model, 
    tokenizer
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Generate the synthetic text corpus in batches.
    
    Args:
        config (GenerationConfig): Pipeline settings.
        prompts (List[Tuple[str, str]]): List of (doc_id, prompt_text).
        model: Loaded Anchor model.
        tokenizer: Loaded tokenizer.
        
    Returns:
        Tuple[List[Dict], float]: The generated dataset records and total generation time.
    """
    log.info("Starting batched generation on device: %s", config.device)
    log.info("Total documents to generate: %d", len(prompts))
    log.info("Batch size: %d", config.batch_size)
    
    dataset_records = []
    start_time = time.time()
    
    gen_params = {
        "temperature": config.temperature,
        "top_k": config.top_k,
        "top_p": config.top_p,
        "repetition_penalty": config.repetition_penalty,
        "max_new_tokens": config.max_new_tokens,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # Process in batches
    for i in tqdm(range(0, len(prompts), config.batch_size), desc="Generating"):
        batch_tuples = prompts[i : i + config.batch_size]
        batch_source_ids = [t[0] for t in batch_tuples]
        batch_prompts = [t[1] for t in batch_tuples]
        
        # Tokenize prompts
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(config.device)
        attention_mask = inputs["attention_mask"].to(config.device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_params
            )
            
        # Decode and structure the output
        for j, (source_id, prompt_text) in enumerate(zip(batch_source_ids, batch_prompts)):
            # Get the exact generated tokens (excluding prompt)
            prompt_len = input_ids[j].shape[0]
            
            # The model output includes the prompt. Let's decode the whole thing.
            full_text = tokenizer.decode(output_ids[j], skip_special_tokens=True)
            
            # Count how many new tokens were actually generated
            # output_ids[j] includes padding on the left if tokenizer did left padding, 
            # but usually it's right padding. Wait, generate automatically handles padding.
            actual_new_tokens = output_ids[j].shape[0] - prompt_len
            
            record = {
                "document_id": str(uuid.uuid4()),
                "prompt": prompt_text,
                "generated_text": full_text,
                "token_count": actual_new_tokens,
                "generation_parameters": {
                    "temperature": config.temperature,
                    "top_k": config.top_k,
                    "top_p": config.top_p,
                    "repetition_penalty": config.repetition_penalty,
                },
                "anchor_version": config.anchor_version,
                "generation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "metadata": {
                    "source_document_id": source_id
                }
            }
            dataset_records.append(record)
            
    total_time = time.time() - start_time
    log.info("Generation complete in %.2f seconds.", total_time)
    
    return dataset_records, total_time
