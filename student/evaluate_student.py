"""
================================================================================
student/evaluate_student.py
================================================================================

Evaluation logic for the Student Model on the synthetic validation dataset.
"""

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

@torch.no_grad()
def evaluate(
    model: PreTrainedModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Compute mean validation loss in evaluation mode.
    Disables gradient tracking to save memory and compute.
    """
    model.eval()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        total_loss += outputs.loss.item()

    mean_loss = total_loss / max(len(loader), 1)
    return mean_loss
