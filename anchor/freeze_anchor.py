"""
================================================================================
anchor/freeze_anchor.py
================================================================================

PART 6 — Freeze Anchor Model
Project: Anchor-Regularized Model Collapse Prevention

Purpose:
    Load the best fine-tuned Anchor Model checkpoint, freeze all of its
    parameters by setting requires_grad = False, and save it as a distinct
    frozen model. This ensures the Anchor distribution remains strictly fixed
    during the self-training curriculum.

Deep Learning Concept — Freezing Parameters:
    In PyTorch, `requires_grad` is a boolean attribute on every Tensor that
    tells the autograd engine whether to track operations and compute
    gradients for that tensor during the backward pass.

    By setting `requires_grad = False` on all parameters:
        1. The weights cannot be updated by any optimizer.
        2. PyTorch saves memory and compute because it doesn't need to build
           the backward computational graph for these layers.
        3. The Anchor Model becomes a deterministic function that maps input
           sequences to fixed reference logits/hidden states.

    This is critical for regularisation against Model Collapse: the Anchor
    must provide a stable, unchanging signal representing the human data
    distribution.

Authors: Deepak (Research Lead)
Created: 2026-07-08
License: MIT
================================================================================
"""

import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Tuple


# ---------------------------------------------------------------------------
# UTF-8 stdout
# ---------------------------------------------------------------------------
def _reconfigure_stdout_utf8() -> None:
    """Force UTF-8 on Windows terminals."""
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )

_reconfigure_stdout_utf8()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
except ImportError as e:
    print(f"[FATAL] Missing dependency: {e}")
    print("Install with:  pip install torch transformers")
    sys.exit(1)

# ============================================================
# CONSTANTS & PATHS
# ============================================================

_THIS_DIR: Path = Path(__file__).resolve().parent
_PROJECT_DIR: Path = _THIS_DIR.parent
CHECKPOINTS_DIR: Path = _PROJECT_DIR / "checkpoints" / "anchor_model"
BEST_CHECKPOINT_DIR: Path = CHECKPOINTS_DIR / "best"
FROZEN_ANCHOR_DIR: Path = CHECKPOINTS_DIR / "frozen"

# ============================================================
# LOGGING
# ============================================================

def _build_logger() -> logging.Logger:
    logger = logging.getLogger("freeze_anchor")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.INFO)
    h.setFormatter(fmt)
    logger.addHandler(h)
    return logger

log: logging.Logger = _build_logger()

# ============================================================
# MODEL LOADING & FREEZING
# ============================================================

def load_best_anchor(checkpoint_dir: Path) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, dict]:
    """
    Load the best fine-tuned Anchor Model and Tokenizer.

    Args:
        checkpoint_dir: Directory containing the saved model and checkpoint_meta.json.

    Returns:
        Tuple of (model, tokenizer, metadata_dict).

    Raises:
        FileNotFoundError: If the checkpoint directory doesn't exist.
    """
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found at: {checkpoint_dir}\n"
            f"Please run  python anchor/train_anchor.py  first."
        )

    log.info("Loading best Anchor Model from: %s", checkpoint_dir)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
        
        meta_path = checkpoint_dir / "checkpoint_meta.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                log.info("Loaded metadata: epoch=%s, val_loss=%s", 
                         meta.get("epoch", "?"), meta.get("val_loss", "?"))
        else:
            log.warning("No checkpoint_meta.json found.")
            
        return model, tokenizer, meta
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from checkpoint: {exc}") from exc


def freeze_model(model: PreTrainedModel) -> None:
    """
    Freeze all parameters in the model.

    Iterates through all named parameters and sets requires_grad = False.
    This guarantees that the PyTorch autograd engine will not compute
    gradients for these tensors, making the model immutable during training.

    Args:
        model: The model to freeze in-place.
    """
    log.info("Freezing all model parameters...")
    frozen_count = 0
    total_count = 0
    
    for name, param in model.named_parameters():
        total_count += 1
        if param.requires_grad:
            param.requires_grad = False
            frozen_count += 1
            
    log.info("Froze %d / %d parameter tensors.", frozen_count, total_count)


def verify_frozen(model: PreTrainedModel) -> bool:
    """
    Verify that no parameter in the model requires gradients.

    Args:
        model: The model to verify.

    Returns:
        bool: True if completely frozen, False otherwise.
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            log.error("Verification failed: parameter '%s' still requires grad!", name)
            return False
    return True

# ============================================================
# SAVING
# ============================================================

def save_frozen_anchor(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizerBase,
    meta: dict,
    out_dir: Path
) -> None:
    """
    Save the frozen model, tokenizer, and metadata to the output directory.

    Args:
        model: The frozen model.
        tokenizer: The associated tokenizer.
        meta: Metadata dictionary from the original checkpoint.
        out_dir: Destination directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Saving frozen Anchor Model to: %s", out_dir)
    
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    
    # Update metadata to indicate it's frozen
    meta["is_frozen"] = True
    meta["frozen_timestamp"] = getattr(time, "time", lambda: 0)()
    
    with open(out_dir / "frozen_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    log.info("Frozen model saved successfully.")

# ============================================================
# DISPLAY HELPERS
# ============================================================

def _section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}\n")

# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Execute the Anchor Model freezing pipeline.

    Steps:
        1. Load the best checkpoint from training.
        2. Freeze all parameters (requires_grad = False).
        3. Verify freezing was successful.
        4. Save as a distinct frozen model.
    """
    _section("Freeze Anchor Model Pipeline")

    # ── Step 1: Load ───────────────────────────────────────────────────
    try:
        model, tokenizer, meta = load_best_anchor(BEST_CHECKPOINT_DIR)
    except FileNotFoundError as e:
        log.error(e)
        sys.exit(1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded         : {model.__class__.__name__}")
    print(f"  Total parameters     : {n_params:,}")
    print(f"  Source checkpoint    : {BEST_CHECKPOINT_DIR}")
    if "val_loss" in meta:
        print(f"  Source val_loss      : {meta['val_loss']}")

    # ── Step 2: Freeze ─────────────────────────────────────────────────
    _section("Freezing Parameters")
    freeze_model(model)

    # ── Step 3: Verify ─────────────────────────────────────────────────
    _section("Verification")
    print("  Checking requires_grad status on all tensors...")
    is_frozen = verify_frozen(model)
    
    if is_frozen:
        print("  [PASS] All parameters successfully frozen (requires_grad=False).")
    else:
        print("  [FAIL] Some parameters are not frozen!")
        sys.exit(1)

    # ── Step 4: Save ───────────────────────────────────────────────────
    _section("Saving Frozen Model")
    save_frozen_anchor(model, tokenizer, meta, FROZEN_ANCHOR_DIR)
    
    print()
    print("  Success! The Anchor Model is now permanently frozen.")
    print(f"  Frozen model path: {FROZEN_ANCHOR_DIR}")
    print("  Next step: run  python anchor/verify_anchor.py")
    print()


if __name__ == "__main__":
    main()
