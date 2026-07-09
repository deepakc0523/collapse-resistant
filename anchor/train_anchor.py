"""
================================================================================
anchor/train_anchor.py
================================================================================

PART 5 — Anchor Model Training
Project: Anchor-Regularized Model Collapse Prevention

Purpose:
    Fine-tune DistilGPT2 on the cleaned WikiText-103 corpus to produce the
    Anchor Model — a frozen reference that captures the human language
    distribution.  This model will later be used to regularise student models
    and prevent mode collapse during iterative self-training.

Deep Learning Concepts:

    CAUSAL LANGUAGE MODELING (CLM):
        At every position i, the model predicts token t_{i+1} from context
        [t_0 ... t_i].  Loss = mean cross-entropy over all real (non-padded)
        token positions.  This is the standard GPT pretraining objective.

    ADAMW OPTIMIZER:
        AdaM + decoupled Weight decay.  Weight decay (L2 regularisation) is
        applied to the raw weights — not to the adaptive gradient moments —
        which corrects the original Adam implementation.

    LINEAR LR SCHEDULER WITH WARMUP:
        Learning rate increases linearly from 0 to peak_lr over *warmup_steps*
        then decreases linearly back to 0.  Warmup stabilises early training
        when gradients are noisy and the model is far from its optimum.

    GRADIENT CLIPPING:
        Clamps the L2 norm of all gradient tensors to max_norm=1.0.
        Prevents exploding gradients — a common failure mode for deep
        Transformer training when the loss spikes.

    MIXED PRECISION (AMP):
        Runs forward/backward passes in float16, but keeps master weights in
        float32.  Reduces GPU memory ~2x and speeds up matrix multiplications
        on Tensor Core GPUs.  GradScaler dynamically rescales the loss to
        prevent float16 underflow during backpropagation.

    CHECKPOINT SAVING:
        We save the model with the lowest validation loss ("best checkpoint").
        Overfitting is detected as val_loss rising while train_loss falls.

Authors: Deepak (Research Lead)
Created: 2026-07-08
License: MIT
================================================================================
"""

import io
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    from torch import Tensor
    from torch.cuda.amp import GradScaler, autocast
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, random_split
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
        PreTrainedModel,
    )
except ImportError as e:
    print(f"[FATAL] Missing dependency: {e}")
    print("Install with:  pip install torch transformers")
    sys.exit(1)

# Add data/ to path so we can import lm_dataset
_ANCHOR_DIR: Path = Path(__file__).resolve().parent        # anchor/
_PROJECT_DIR: Path = _ANCHOR_DIR.parent                    # project root
_DATASETS_DIR: Path = _PROJECT_DIR / "data"
sys.path.insert(0, str(_DATASETS_DIR))

try:
    from lm_dataset import WikiTextLMDataset, TOKENIZED_DIR
except ImportError as e:
    print(f"[FATAL] Could not import lm_dataset: {e}")
    print("Run  python data/tokenize_dataset.py  first.")
    sys.exit(1)

# ============================================================
# CONSTANTS / HYPERPARAMETERS
# ============================================================

MODEL_NAME: str = "distilgpt2"

# --- Training hyperparameters ---
BATCH_SIZE: int        = 8          # sequences per gradient step
GRAD_ACCUM_STEPS: int  = 4          # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS = 32
NUM_EPOCHS: int        = 1          # full passes over the corpus
LEARNING_RATE: float   = 5e-5       # peak LR for AdamW
WEIGHT_DECAY: float    = 0.01       # L2 regularisation coefficient
WARMUP_RATIO: float    = 0.06       # fraction of steps used for LR warmup
MAX_GRAD_NORM: float   = 1.0        # gradient clipping threshold
VAL_SPLIT_RATIO: float = 0.02       # fraction of data reserved for validation
MAX_SAMPLES: Optional[int] = 10000   # None = use full dataset; set e.g. 5000 for fast debug
NUM_WORKERS: int       = 0          # DataLoader workers (0 = safe on Windows)

# --- Mixed precision ---
USE_AMP: bool = torch.cuda.is_available()  # AMP only beneficial on GPU

# --- Paths ---
CHECKPOINT_DIR: Path = _PROJECT_DIR / "checkpoints" / "anchor_model"
LOG_FILE: Path        = CHECKPOINT_DIR / "training_log.txt"

# ============================================================
# LOGGING
# ============================================================

def _build_logger() -> logging.Logger:
    """Logger that writes to stdout AND the training log file."""
    logger = logging.getLogger("train_anchor")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger

log: logging.Logger = _build_logger()


def _add_file_handler(path: Path) -> None:
    """Attach a file handler to the logger once the checkpoint dir exists."""
    if any(isinstance(h, logging.FileHandler) for h in log.handlers):
        return
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)

# ============================================================
# DEVICE SETUP
# ============================================================

def get_device() -> torch.device:
    """
    Select the best available compute device.

    Priority: CUDA (GPU) > MPS (Apple Silicon) > CPU

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("Device: CUDA (%s)", torch.cuda.get_device_name(0))
        log.info("  GPU memory: %.1f GB total / %.1f GB free",
                 torch.cuda.get_device_properties(0).total_memory / 1e9,
                 (torch.cuda.get_device_properties(0).total_memory
                  - torch.cuda.memory_allocated(0)) / 1e9)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Device: Apple MPS")
    else:
        device = torch.device("cpu")
        log.warning("Device: CPU — training will be slow. Consider a GPU.")
    return device

# ============================================================
# DATA LOADING
# ============================================================

def build_dataloaders(
    batch_size: int = BATCH_SIZE,
    val_ratio: float = VAL_SPLIT_RATIO,
    max_samples: Optional[int] = MAX_SAMPLES,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and validation DataLoaders.

    The tokenized dataset is split as:
        val_size  = int(len(dataset) * val_ratio)
        train_size = len(dataset) - val_size

    Using random_split ensures no data leakage between the two loaders.

    Args:
        batch_size:  Sequences per batch.
        val_ratio:   Fraction of data for validation.
        max_samples: Optional sample cap (None = full dataset).

    Returns:
        Tuple (train_loader, val_loader).
    """
    log.info("Loading dataset from: %s", TOKENIZED_DIR)
    full_dataset = WikiTextLMDataset(data_dir=TOKENIZED_DIR, max_samples=max_samples)
    n_total = len(full_dataset)

    n_val   = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    log.info("Split: train=%d, val=%d  (total=%d)", n_train, n_val, n_total)

    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,     # keep all batches the same size for stable training
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    log.info("DataLoaders: train=%d batches, val=%d batches.", len(train_loader), len(val_loader))
    return train_loader, val_loader

# ============================================================
# MODEL SETUP
# ============================================================

def load_model(
    model_name: str = MODEL_NAME,
    device: Optional[torch.device] = None,
) -> PreTrainedModel:
    """
    Load DistilGPT2 and move it to the target device.

    DistilGPT2 is used as the starting point for the Anchor Model because:
        1. It is pretrained on a diverse web corpus (WebText), giving it
           a strong prior over English language patterns.
        2. It is small enough (82 M parameters) to fine-tune quickly on
           WikiText-103 without requiring multi-GPU infrastructure.
        3. Fine-tuning on WikiText-103 specialises the model to the
           high-quality, human-edited Wikipedia distribution — exactly
           the reference distribution the Anchor needs to embody.

    Args:
        model_name: HuggingFace model identifier.
        device:     Target device.

    Returns:
        PreTrainedModel on the specified device.
    """
    log.info("Loading model: %s", model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except Exception as exc:
        raise RuntimeError(f"Model load failed: {exc}") from exc

    if device is not None:
        model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model loaded: %d M parameters", n_params // 1_000_000)
    return model

# ============================================================
# OPTIMIZER & SCHEDULER
# ============================================================

def build_optimizer_scheduler(
    model: PreTrainedModel,
    n_train_steps: int,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    warmup_ratio: float = WARMUP_RATIO,
) -> Tuple:
    """
    Build AdamW optimizer and linear warmup-decay LR scheduler.

    AdamW parameter groups:
        - Weight matrices & embeddings: apply weight decay (regularise).
        - Biases & LayerNorm params:    NO weight decay (these are scale
          parameters — decaying them degrades normalisation).

    Args:
        model:          The DistilGPT2 model.
        n_train_steps:  Total number of optimizer steps.
        lr:             Peak learning rate.
        weight_decay:   L2 coefficient.
        warmup_ratio:   Fraction of steps for linear warmup.

    Returns:
        (optimizer, scheduler)
    """
    # Separate params: no decay for biases and LayerNorm weights
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
    log.info("Optimizer: AdamW  lr=%.2e  weight_decay=%.4f", lr, weight_decay)
    log.info("Scheduler: linear warmup (%d steps) -> linear decay (%d steps)",
             warmup_steps, n_train_steps - warmup_steps)
    return optimizer, scheduler

# ============================================================
# TRAINING LOOP
# ============================================================

def train_one_epoch(
    model: PreTrainedModel,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    scaler: Optional[GradScaler],
    device: torch.device,
    epoch: int,
    grad_accum_steps: int = GRAD_ACCUM_STEPS,
    max_grad_norm: float = MAX_GRAD_NORM,
) -> float:
    """
    Run one full training epoch.

    Gradient accumulation:
        We accumulate gradients over *grad_accum_steps* batches before calling
        optimizer.step().  This simulates a larger effective batch size without
        requiring more GPU memory.

    Args:
        model:            DistilGPT2 model in training mode.
        loader:           Training DataLoader.
        optimizer:        AdamW optimizer.
        scheduler:        LR scheduler.
        scaler:           GradScaler for AMP (None on CPU).
        device:           Compute device.
        epoch:            Current epoch index (0-based, for logging).
        grad_accum_steps: Number of micro-batches before optimizer step.
        max_grad_norm:    Gradient clipping threshold.

    Returns:
        float: Mean training loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    n_steps    = 0
    t_start    = time.time()

    optimizer.zero_grad()

    for step_idx, batch in enumerate(loader):
        # Move tensors to device
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # Forward pass (optionally in float16 AMP)
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

        total_loss += loss.item() * grad_accum_steps   # undo the division for logging

        # Optimizer step after accumulating enough micro-batches
        if (step_idx + 1) % grad_accum_steps == 0:
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

            # Logging every 100 optimizer steps
            if n_steps % 100 == 0:
                elapsed = time.time() - t_start
                current_lr = scheduler.get_last_lr()[0]
                avg_loss   = total_loss / (step_idx + 1)
                gpu_mem    = ""
                if torch.cuda.is_available():
                    gpu_mem = f"  GPU:{torch.cuda.memory_allocated() / 1e9:.1f}GB"
                log.info(
                    "Epoch %d | step %d/%d | loss=%.4f | lr=%.2e | t=%.0fs%s",
                    epoch + 1, step_idx + 1, len(loader),
                    avg_loss, current_lr, elapsed, gpu_mem,
                )

    mean_loss = total_loss / max(len(loader), 1)
    return mean_loss


@torch.no_grad()
def evaluate(
    model: PreTrainedModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Compute mean validation loss in evaluation mode.

    @torch.no_grad() disables gradient tracking, saving memory and compute
    during inference — we never backpropagate through validation data.

    Args:
        model:   DistilGPT2 model.
        loader:  Validation DataLoader.
        device:  Compute device.

    Returns:
        float: Mean validation loss.  Also computes perplexity = exp(loss).
    """
    model.eval()
    total_loss = 0.0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()

    mean_loss = total_loss / max(len(loader), 1)
    return mean_loss

# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================

def save_checkpoint(
    model: PreTrainedModel,
    tokenizer,
    epoch: int,
    val_loss: float,
    checkpoint_dir: Path,
    tag: str = "best",
) -> None:
    """
    Save a model checkpoint to disk.

    Saves:
        - Model weights (HuggingFace format: config.json + pytorch_model.bin)
        - Tokenizer files
        - checkpoint_meta.json with epoch and loss

    Args:
        model:          Trained model.
        tokenizer:      Associated tokenizer.
        epoch:          Epoch number (1-based).
        val_loss:       Validation loss at this checkpoint.
        checkpoint_dir: Root directory for checkpoints.
        tag:            Sub-directory tag ("best" or f"epoch_{epoch}").
    """
    save_path = checkpoint_dir / tag
    save_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    meta = {
        "epoch":    epoch,
        "val_loss": round(val_loss, 6),
        "perplexity": round(math.exp(val_loss), 4),
        "model_name": MODEL_NAME,
    }
    with open(save_path / "checkpoint_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info("Checkpoint saved: %s  (epoch=%d  val_loss=%.4f  ppl=%.2f)",
             save_path, epoch, val_loss, math.exp(val_loss))

# ============================================================
# DISPLAY HELPERS
# ============================================================

def _section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}\n")


def print_epoch_summary(
    epoch: int,
    n_epochs: int,
    train_loss: float,
    val_loss: float,
    lr: float,
    elapsed: float,
    is_best: bool,
) -> None:
    """Log and print a single-line epoch summary."""
    ppl = math.exp(val_loss)
    best_tag = " <-- BEST" if is_best else ""
    print(
        f"  Epoch {epoch:>2}/{n_epochs} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"ppl={ppl:.2f} | "
        f"lr={lr:.2e} | "
        f"t={elapsed:.0f}s{best_tag}"
    )
    log.info(
        "Epoch %d/%d | train=%.4f | val=%.4f | ppl=%.2f | lr=%.2e | t=%.0fs%s",
        epoch, n_epochs, train_loss, val_loss, ppl, lr, elapsed, best_tag,
    )

# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main() -> None:
    """
    Full Anchor Model training pipeline.

    Steps:
        1.  Setup directories and file logging.
        2.  Select compute device.
        3.  Load training and validation DataLoaders.
        4.  Load DistilGPT2 model and tokenizer.
        5.  Build AdamW optimizer and LR scheduler.
        6.  Optionally set up AMP GradScaler.
        7.  Training loop (NUM_EPOCHS epochs):
              a. Train one epoch with gradient accumulation + clipping.
              b. Evaluate on validation set.
              c. Save checkpoint if val_loss improved.
              d. Print epoch summary.
        8.  Save the final model.
        9.  Print the training summary.
    """
    _section("Anchor Model Training — DistilGPT2 on WikiText-103")

    # ── Step 1: Directories + file log ────────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    _add_file_handler(LOG_FILE)
    log.info("Checkpoint directory: %s", CHECKPOINT_DIR)
    log.info("Log file: %s", LOG_FILE)

    # ── Step 2: Device ─────────────────────────────────────────────────
    device = get_device()

    # ── Step 3: DataLoaders ────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(BATCH_SIZE, VAL_SPLIT_RATIO, MAX_SAMPLES)

    # ── Step 4: Model + Tokenizer ──────────────────────────────────────
    model     = load_model(MODEL_NAME, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Step 5: Optimizer + Scheduler ─────────────────────────────────
    total_opt_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
    optimizer, scheduler = build_optimizer_scheduler(
        model, total_opt_steps, LEARNING_RATE, WEIGHT_DECAY, WARMUP_RATIO
    )
    log.info("Total optimizer steps: %d", total_opt_steps)

    # ── Step 6: AMP scaler ─────────────────────────────────────────────
    scaler: Optional[GradScaler] = GradScaler() if USE_AMP else None
    log.info("Mixed precision (AMP): %s", "enabled" if scaler else "disabled (CPU)")

    # Print training config
    _section("Training Configuration")
    print(f"  Model             : {MODEL_NAME}")
    print(f"  Epochs            : {NUM_EPOCHS}")
    print(f"  Batch size        : {BATCH_SIZE} (micro) x {GRAD_ACCUM_STEPS} accum = {BATCH_SIZE * GRAD_ACCUM_STEPS} effective")
    print(f"  Learning rate     : {LEARNING_RATE:.2e}")
    print(f"  Weight decay      : {WEIGHT_DECAY}")
    print(f"  Warmup ratio      : {WARMUP_RATIO}")
    print(f"  Gradient clipping : {MAX_GRAD_NORM}")
    print(f"  Mixed precision   : {USE_AMP}")
    print(f"  Device            : {device}")
    print(f"  Train batches     : {len(train_loader)}")
    print(f"  Val batches       : {len(val_loader)}")
    print(f"  Total opt steps   : {total_opt_steps}")
    print()

    # ── Step 7: Training loop ──────────────────────────────────────────
    _section("Training Loop")
    best_val_loss = float("inf")
    training_history = []
    t_train_start = time.time()

    for epoch in range(NUM_EPOCHS):
        t_epoch_start = time.time()

        # --- Train ---
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, GRAD_ACCUM_STEPS, MAX_GRAD_NORM,
        )

        # --- Validate ---
        val_loss = evaluate(model, val_loader, device)

        # --- Checkpoint ---
        current_lr = scheduler.get_last_lr()[0]
        is_best    = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            save_checkpoint(model, tokenizer, epoch + 1, val_loss,
                            CHECKPOINT_DIR, tag="best")

        # Always save per-epoch checkpoint
        save_checkpoint(model, tokenizer, epoch + 1, val_loss,
                        CHECKPOINT_DIR, tag=f"epoch_{epoch + 1}")

        elapsed = time.time() - t_epoch_start
        print_epoch_summary(epoch + 1, NUM_EPOCHS, train_loss, val_loss,
                             current_lr, elapsed, is_best)

        training_history.append({
            "epoch":      epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "perplexity": round(math.exp(val_loss), 4),
            "lr":         current_lr,
            "elapsed_s":  round(elapsed, 1),
        })

    total_elapsed = time.time() - t_train_start

    # ── Step 8: Save final model ───────────────────────────────────────
    save_checkpoint(model, tokenizer, NUM_EPOCHS, val_loss,
                    CHECKPOINT_DIR, tag="final")

    # Save training history
    history_path = CHECKPOINT_DIR / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)

    # ── Step 9: Summary ────────────────────────────────────────────────
    _section("Training Complete")
    print(f"  Total time        : {total_elapsed / 60:.1f} min")
    print(f"  Best val loss     : {best_val_loss:.4f}  (ppl={math.exp(best_val_loss):.2f})")
    print(f"  Best checkpoint   : {CHECKPOINT_DIR / 'best'}")
    print(f"  Training log      : {LOG_FILE}")
    print(f"  History JSON      : {history_path}")
    print()
    print("  Next step: python anchor/freeze_anchor.py")
    log.info("Training complete. Total time: %.1f min. Best val_loss: %.4f",
             total_elapsed / 60, best_val_loss)


if __name__ == "__main__":
    main()
