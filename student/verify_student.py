"""
================================================================================
student/verify_student.py
================================================================================

Verification script to ensure all research constraints for the Student Model
are met before full training execution.

Checks:
1. Model is initialized randomly (no pretrained weights).
2. Data loads correctly from the synthetic dataset.
3. Forward pass executes and loss is calculated.
4. Gradients flow correctly during a dummy backward pass.
"""

import sys
import torch
from student_config import SYNTHETIC_DATA_DIR, MODEL_TYPE
from student_model import load_random_student_model, get_tokenizer
from synthetic_dataset import build_dataloaders
from transformers import AutoModelForCausalLM

def verify_random_initialization():
    print("Verification 1: Random Initialization")
    
    student = load_random_student_model(MODEL_TYPE)
    pretrained = AutoModelForCausalLM.from_pretrained(MODEL_TYPE)
    
    # Check if a sample of weights are different
    sample_name = "transformer.h.0.attn.c_attn.weight"
    student_weight = student.state_dict()[sample_name]
    pretrained_weight = pretrained.state_dict()[sample_name]
    
    if torch.allclose(student_weight, pretrained_weight):
        print("[FAIL] Student model weights match pretrained weights!")
        sys.exit(1)
    else:
        print("[PASS] Student model is initialized randomly.")

def verify_synthetic_data():
    print("\nVerification 2: Synthetic Data Loading")
    try:
        train_loader, _ = build_dataloaders(SYNTHETIC_DATA_DIR, batch_size=2, val_ratio=0.1, max_samples=100)
        batch = next(iter(train_loader))
        
        if "input_ids" not in batch or "labels" not in batch:
            print("[FAIL] Missing required keys in batch.")
            sys.exit(1)
            
        print(f"[PASS] Successfully loaded batch of shape: {batch['input_ids'].shape}")
        return train_loader
    except Exception as e:
        print(f"[FAIL] Error loading synthetic data: {e}")
        sys.exit(1)

def verify_forward_backward(train_loader):
    print("\nVerification 3 & 4: Forward Pass & Gradient Flow")
    
    model = load_random_student_model(MODEL_TYPE)
    model.train()
    
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    
    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    
    if loss is None or torch.isnan(loss):
        print("[FAIL] Forward pass failed or returned NaN loss.")
        sys.exit(1)
    
    print(f"[PASS] Forward pass successful. Initial Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_grad = True
            break
            
    if not has_grad:
        print("[FAIL] No gradients found after backward pass.")
        sys.exit(1)
        
    print("[PASS] Backward pass successful. Gradients are flowing.")

def main():
    print("==================================================")
    print("STUDENT PIPELINE VERIFICATION")
    print("==================================================\n")
    
    verify_random_initialization()
    loader = verify_synthetic_data()
    verify_forward_backward(loader)
    
    print("\n==================================================")
    print("ALL VERIFICATIONS PASSED")
    print("==================================================")

if __name__ == "__main__":
    main()
