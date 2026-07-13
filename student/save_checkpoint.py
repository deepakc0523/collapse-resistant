"""
================================================================================
student/save_checkpoint.py
================================================================================

Checkpoint management for the Student Model.
"""

import json
import math
import logging
from pathlib import Path
from transformers import PreTrainedModel, PreTrainedTokenizer

def save_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    epoch: int,
    val_loss: float,
    checkpoint_dir: Path,
    model_name: str,
    logger: logging.Logger,
    tag: str = "best",
) -> None:
    """
    Save the model, tokenizer, and metadata to the specified checkpoint directory.
    """
    save_path = checkpoint_dir / tag
    save_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    meta = {
        "epoch": epoch,
        "val_loss": round(val_loss, 6),
        "perplexity": round(math.exp(val_loss), 4),
        "model_name": model_name,
    }
    
    with open(save_path / "checkpoint_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Checkpoint saved: %s  (epoch=%d  val_loss=%.4f  ppl=%.2f)",
             save_path, epoch, val_loss, math.exp(val_loss))
