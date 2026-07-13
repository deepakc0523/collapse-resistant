"""
================================================================================
student/statistics.py
================================================================================

Handles logging model statistics (parameters) and saving the training history.
"""

import json
from pathlib import Path
import logging
from transformers import PreTrainedModel

def save_model_statistics(
    model: PreTrainedModel,
    save_path: Path,
    logger: logging.Logger
) -> None:
    """
    Saves statistics about the model, particularly parameter counts, to verify
    it matches the expected architecture.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    stats = {
        "architecture": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "parameter_breakdown": {
            name: p.numel() for name, p in model.named_parameters()
        }
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Model statistics saved to %s", save_path)
