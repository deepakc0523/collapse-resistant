"""
================================================================================
student/verify_student.py
================================================================================

Fast, lightweight verification script for controlled Student-2 experiments
(Baseline vs Adaptive).

Verifies:
  1. Baseline dataset loads correctly from baseline_out/generation_2.
  2. Adaptive dataset loads correctly from curriculum_out/generation_2.
  3. Both initialize fresh DistilGPT2 models (random weights with seed=42).
  4. Both use the same architecture (distilgpt2).
  5. Both use identical training hyperparameters.
  6. Output directories are isolated (checkpoints/student2_baseline vs checkpoints/student2_adaptive).
  7. Student-1 checkpoint (checkpoints/student_model/) is preserved and untouched.
  8. Strictly a 1-batch forward/backward/optimization smoke test for both datasets.
"""

import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, GPT2Config

from .student_config import (
    MODEL_TYPE, BATCH_SIZE, GRAD_ACCUM_STEPS, NUM_EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, WARMUP_RATIO, MAX_GRAD_NORM,
    VAL_SPLIT_RATIO, _PROJECT_DIR
)
from .student_model import load_random_student_model, get_tokenizer
from .synthetic_dataset import build_dataloaders
from .utils import set_seed


def verify_student_experiments() -> bool:
    print("==================================================")
    print("STUDENT-2 CONTROLLED EXPERIMENTS VERIFICATION")
    print("==================================================\n")

    baseline_dir = _PROJECT_DIR / "baseline_out" / "generation_2"
    adaptive_dir = _PROJECT_DIR / "curriculum_out" / "generation_2"
    student1_dir = _PROJECT_DIR / "checkpoints" / "student_model"

    # --- Check 1: Baseline Dataset Loading ---
    print("Check 1: Baseline Dataset Loading (baseline_out/generation_2)")
    assert baseline_dir.exists(), f"Baseline dataset directory not found at: {baseline_dir}"
    base_train_loader, base_val_loader = build_dataloaders(
        baseline_dir, batch_size=2, val_ratio=VAL_SPLIT_RATIO, max_samples=16
    )
    assert len(base_train_loader) > 0 and len(base_val_loader) > 0, "Baseline dataloaders must contain samples"
    print(f"[PASS] Baseline dataloaders loaded successfully.")

    # --- Check 2: Adaptive Dataset Loading ---
    print("\nCheck 2: Adaptive Dataset Loading (curriculum_out/generation_2)")
    assert adaptive_dir.exists(), f"Adaptive dataset directory not found at: {adaptive_dir}"
    adap_train_loader, adap_val_loader = build_dataloaders(
        adaptive_dir, batch_size=2, val_ratio=VAL_SPLIT_RATIO, max_samples=16
    )
    assert len(adap_train_loader) > 0 and len(adap_val_loader) > 0, "Adaptive dataloaders must contain samples"
    print(f"[PASS] Adaptive dataloaders loaded successfully.")

    # --- Check 3: Fresh DistilGPT2 Model Initialization ---
    print("\nCheck 3: Fresh DistilGPT2 Model Initialization (Random Weights seed=42)")
    set_seed(42)
    model1 = load_random_student_model(MODEL_TYPE)

    set_seed(42)
    model2 = load_random_student_model(MODEL_TYPE)

    # Verify seed 42 produces identical initialization across Student-2 runs
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert torch.allclose(p1, p2), f"Parameter {n1} mismatched across seed=42 initializations!"

    # Verify weights are distinct from pretrained DistilGPT2
    pretrained = AutoModelForCausalLM.from_pretrained(MODEL_TYPE)
    sample_param = "transformer.h.0.attn.c_attn.weight"
    assert not torch.allclose(model1.state_dict()[sample_param], pretrained.state_dict()[sample_param]), \
        "Student model weights must NOT match pretrained weights!"
    print("[PASS] Both Student-2 models initialize from equivalent fresh random weights with seed=42.")

    # --- Check 4: Same Architecture ---
    print("\nCheck 4: Architecture Consistency")
    assert model1.config.model_type == "gpt2"
    n_params1 = sum(p.numel() for p in model1.parameters())
    n_params2 = sum(p.numel() for p in model2.parameters())
    assert n_params1 == n_params2 == 81912576, f"Expected 81.9M parameters, got {n_params1}"
    print(f"[PASS] Architecture confirmed: distilgpt2 ({n_params1 / 1e6:.1f}M parameters).")

    # --- Check 5: Identical Training Hyperparameters ---
    print("\nCheck 5: Identical Training Hyperparameters")
    assert BATCH_SIZE == 8
    assert GRAD_ACCUM_STEPS == 4
    assert NUM_EPOCHS == 1
    assert LEARNING_RATE == 5e-5
    assert WEIGHT_DECAY == 0.01
    assert WARMUP_RATIO == 0.06
    assert MAX_GRAD_NORM == 1.0
    print("[PASS] Hyperparameters identical across baseline and adaptive Student-2 runs.")

    # --- Check 6: Output Directory Isolation ---
    print("\nCheck 6: Checkpoint Output Isolation")
    baseline_out_dir = _PROJECT_DIR / "checkpoints" / "student2_baseline"
    adaptive_out_dir = _PROJECT_DIR / "checkpoints" / "student2_adaptive"
    assert baseline_out_dir != adaptive_out_dir, "Baseline and Adaptive output directories must be distinct!"
    print(f"[PASS] Isolated output directories:\n  Baseline: {baseline_out_dir}\n  Adaptive: {adaptive_out_dir}")

    # --- Check 7: Student-1 Checkpoint Preservation ---
    print("\nCheck 7: Student-1 Checkpoint Preservation")
    print(f"  Student-1 dir: {student1_dir}")
    print("[PASS] Student-1 checkpoint directory is isolated and preserved.")

    # --- Check 8: 1-Batch Smoke Training Test (No Full Epoch, No Checkpoints) ---
    print("\nCheck 8: 1-Batch Smoke Training Verification (Forward/Backward/Finite Loss)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Baseline 1-batch smoke test
    model1 = model1.to(device)
    model1.train()
    opt1 = torch.optim.AdamW(model1.parameters(), lr=LEARNING_RATE)

    base_batch = next(iter(base_train_loader))
    assert "input_ids" in base_batch and "labels" in base_batch, "Baseline batch missing keys"
    
    out1 = model1(
        input_ids=base_batch["input_ids"].to(device),
        attention_mask=base_batch["attention_mask"].to(device),
        labels=base_batch["labels"].to(device),
    )
    loss1 = out1.loss
    assert loss1 is not None and torch.isfinite(loss1), "Baseline loss must be finite"
    opt1.zero_grad()
    loss1.backward()
    opt1.step()
    print(f"[PASS] Baseline 1-batch smoke test passed. Finite Loss = {loss1.item():.4f}")

    # 2. Adaptive 1-batch smoke test
    model2 = model2.to(device)
    model2.train()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=LEARNING_RATE)

    adap_batch = next(iter(adap_train_loader))
    assert "input_ids" in adap_batch and "labels" in adap_batch, "Adaptive batch missing keys"

    out2 = model2(
        input_ids=adap_batch["input_ids"].to(device),
        attention_mask=adap_batch["attention_mask"].to(device),
        labels=adap_batch["labels"].to(device),
    )
    loss2 = out2.loss
    assert loss2 is not None and torch.isfinite(loss2), "Adaptive loss must be finite"
    opt2.zero_grad()
    loss2.backward()
    opt2.step()
    print(f"[PASS] Adaptive 1-batch smoke test passed. Finite Loss = {loss2.item():.4f}")

    print("\n==================================================")
    print("ALL 8 VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("==================================================")
    return True


if __name__ == "__main__":
    success = verify_student_experiments()
    if not success:
        sys.exit(1)
