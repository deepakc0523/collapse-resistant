"""
================================================================================
student/run_student_training.py
================================================================================

Main entry point for the Student Model Training Pipeline.
Orchestrates data loading, model initialization (random weights),
and the training/evaluation loop.
"""

import math
import time
import json
import torch
from torch.cuda.amp import GradScaler

# Import components from the student module
from student_config import (
    SYNTHETIC_DATA_DIR, CHECKPOINT_DIR, TRAINING_LOG_FILE,
    TRAINING_HISTORY_FILE, MODEL_STATISTICS_FILE, MODEL_TYPE,
    BATCH_SIZE, GRAD_ACCUM_STEPS, NUM_EPOCHS, LEARNING_RATE,
    WEIGHT_DECAY, WARMUP_RATIO, MAX_GRAD_NORM, VAL_SPLIT_RATIO,
    MAX_SAMPLES, NUM_WORKERS
)
from utils import get_logger, get_device, set_seed
from synthetic_dataset import build_dataloaders
from student_model import load_random_student_model, get_tokenizer
from train_student import build_optimizer_scheduler, train_one_epoch
from evaluate_student import evaluate
from save_checkpoint import save_checkpoint
from student_statistics import save_model_statistics

def _section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}\n")

def main() -> None:
    set_seed(42)
    
    _section("Student Model Training — Recursive Synthetic Learning")

    # 1. Setup Logging
    logger = get_logger("run_student_training", TRAINING_LOG_FILE)
    logger.info("Initializing Student Model Pipeline...")
    logger.info("Target Data Dir: %s", SYNTHETIC_DATA_DIR)
    
    # 2. Device Setup
    device = get_device(logger)
    
    # 3. DataLoaders
    train_loader, val_loader = build_dataloaders(
        data_dir=SYNTHETIC_DATA_DIR,
        batch_size=BATCH_SIZE,
        val_ratio=VAL_SPLIT_RATIO,
        max_samples=MAX_SAMPLES,
        num_workers=NUM_WORKERS
    )
    logger.info("DataLoaders built: %d train batches, %d val batches", len(train_loader), len(val_loader))

    # 4. Model and Tokenizer
    tokenizer = get_tokenizer(MODEL_TYPE)
    model = load_random_student_model(MODEL_TYPE, device, logger)
    save_model_statistics(model, MODEL_STATISTICS_FILE, logger)

    # 5. Optimizer & Scheduler
    total_opt_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
    optimizer, scheduler = build_optimizer_scheduler(
        model, total_opt_steps, LEARNING_RATE, WEIGHT_DECAY, WARMUP_RATIO, logger
    )

    # 6. Mixed Precision
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    logger.info("Mixed precision (AMP): %s", "enabled" if scaler else "disabled (CPU)")

    _section("Training Configuration")
    print(f"  Model             : {MODEL_TYPE} (RANDOM INIT)")
    print(f"  Data Source       : {SYNTHETIC_DATA_DIR}")
    print(f"  Epochs            : {NUM_EPOCHS}")
    print(f"  Batch size        : {BATCH_SIZE} (micro) x {GRAD_ACCUM_STEPS} accum = {BATCH_SIZE * GRAD_ACCUM_STEPS} effective")
    print(f"  Learning rate     : {LEARNING_RATE:.2e}")
    print(f"  Weight decay      : {WEIGHT_DECAY}")
    print(f"  Warmup ratio      : {WARMUP_RATIO}")
    print(f"  Gradient clipping : {MAX_GRAD_NORM}")
    print(f"  Device            : {device}")

    # 7. Training Loop
    _section("Training Loop")
    best_val_loss = float("inf")
    training_history = []
    t_train_start = time.time()

    for epoch in range(NUM_EPOCHS):
        t_epoch_start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, GRAD_ACCUM_STEPS, MAX_GRAD_NORM, logger
        )

        val_loss = evaluate(model, val_loader, device)

        current_lr = scheduler.get_last_lr()[0]
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            save_checkpoint(model, tokenizer, epoch + 1, val_loss, CHECKPOINT_DIR, MODEL_TYPE, logger, tag="best")

        save_checkpoint(model, tokenizer, epoch + 1, val_loss, CHECKPOINT_DIR, MODEL_TYPE, logger, tag=f"epoch_{epoch + 1}")

        elapsed = time.time() - t_epoch_start
        ppl = math.exp(val_loss)
        best_tag = " <-- BEST" if is_best else ""
        
        print(
            f"  Epoch {epoch+1:>2}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"ppl={ppl:.2f} | "
            f"lr={current_lr:.2e} | "
            f"t={elapsed:.0f}s{best_tag}"
        )

        training_history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "perplexity": round(ppl, 4),
            "lr": current_lr,
            "elapsed_s": round(elapsed, 1),
        })

    total_elapsed = time.time() - t_train_start

    # 8. Save Final Model and History
    save_checkpoint(model, tokenizer, NUM_EPOCHS, val_loss, CHECKPOINT_DIR, MODEL_TYPE, logger, tag="final")

    with open(TRAINING_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)

    _section("Training Complete")
    print(f"  Total time        : {total_elapsed / 60:.1f} min")
    print(f"  Best val loss     : {best_val_loss:.4f}  (ppl={math.exp(best_val_loss):.2f})")
    print(f"  Best checkpoint   : {CHECKPOINT_DIR / 'best'}")
    print(f"  History JSON      : {TRAINING_HISTORY_FILE}")
    logger.info("Training complete. Total time: %.1f min. Best val_loss: %.4f", total_elapsed / 60, best_val_loss)

if __name__ == "__main__":
    main()
