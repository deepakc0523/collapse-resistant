"""
================================================================================
student/train_student.py
================================================================================

Core training logic for the Student Model.
Implements the forward pass, loss calculation, backpropagation, and optimization.
Supports Gradient Accumulation, Gradient Clipping, and Mixed Precision (AMP).
"""

import time
import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, get_linear_schedule_with_warmup
import logging
from typing import Tuple, Optional

def build_optimizer_scheduler(
    model: PreTrainedModel,
    n_train_steps: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    logger: logging.Logger
) -> Tuple:
    """
    Build AdamW optimizer and linear warmup-decay LR scheduler.
    Excludes biases and LayerNorm weights from weight decay.
    """
    no_decay = {"bias", "layer_norm.weight", "layernorm.weight", "ln_"}
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n.lower() for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n.lower() for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(param_groups, lr=lr, eps=1e-8)
    warmup_steps = int(n_train_steps * warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=n_train_steps,
    )
    
    logger.info("Optimizer: AdamW  lr=%.2e  weight_decay=%.4f", lr, weight_decay)
    logger.info("Scheduler: linear warmup (%d steps) -> linear decay (%d steps)",
             warmup_steps, n_train_steps - warmup_steps)
             
    return optimizer, scheduler

def train_one_epoch(
    model: PreTrainedModel,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    scaler: Optional[GradScaler],
    device: torch.device,
    epoch: int,
    grad_accum_steps: int,
    max_grad_norm: float,
    logger: logging.Logger
) -> float:
    """
    Run one full training epoch over the synthetic dataset.
    """
    model.train()
    total_loss = 0.0
    n_steps = 0
    t_start = time.time()

    optimizer.zero_grad()

    for step_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass (Mixed Precision optional)
        if scaler is not None:
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / grad_accum_steps
            loss.backward()

        total_loss += loss.item() * grad_accum_steps

        # Optimization step
        if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            n_steps += 1

            if n_steps % 100 == 0:
                elapsed = time.time() - t_start
                current_lr = scheduler.get_last_lr()[0]
                avg_loss = total_loss / (step_idx + 1)
                gpu_mem = f"  GPU:{torch.cuda.memory_allocated() / 1e9:.1f}GB" if torch.cuda.is_available() else ""
                
                logger.info(
                    "Epoch %d | step %d/%d | loss=%.4f | lr=%.2e | t=%.0fs%s",
                    epoch + 1, step_idx + 1, len(loader),
                    avg_loss, current_lr, elapsed, gpu_mem,
                )

    mean_loss = total_loss / max(len(loader), 1)
    return mean_loss
