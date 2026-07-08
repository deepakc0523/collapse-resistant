"""
================================================================================
anchor/verify_anchor.py
================================================================================

PART 7 — Verify Anchor Model
Project: Anchor-Regularized Model Collapse Prevention

Purpose:
    Perform a comprehensive final verification of the frozen Anchor Model.
    This script loads the frozen model, confirms its weights are locked,
    runs a forward pass to inspect hidden states and logits, and generates
    text to verify its language modeling capabilities.

Deep Learning Concepts:

    VERIFYING FROZEN STATE:
        We explicitly check that `requires_grad == False` for every parameter.
        We also verify the model is in evaluation mode (`model.training == False`),
        which disables dropout and locks batch normalization (though GPT uses
        LayerNorm, eval mode is still standard practice for inference).

    HIDDEN STATES:
        By passing `output_hidden_states=True` to the forward pass, we can
        extract the contextualized embeddings produced by every layer of the
        Transformer. These intermediate representations are crucial for some
        advanced regularization techniques (like feature-matching).

    LOGITS & PREDICTION:
        The final layer outputs raw logits (unnormalized scores for each token
        in the vocabulary). We apply softmax to get probabilities and use
        `argmax` or sampling to select the next token.

    GENERATION:
        We use Hugging Face's `generate()` method to autoregressively sample
        new text, proving the model has learned a coherent language distribution
        from WikiText-103.

Authors: Deepak (Research Lead)
Created: 2026-07-08
License: MIT
================================================================================
"""

import io
import logging
import sys
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
FROZEN_ANCHOR_DIR: Path = _PROJECT_DIR / "checkpoints" / "anchor_model" / "frozen"
TEST_PROMPT: str = "The architecture of a deep neural network"

# ============================================================
# LOGGING
# ============================================================

def _build_logger() -> logging.Logger:
    logger = logging.getLogger("verify_anchor")
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
# LOAD MODEL
# ============================================================

def load_frozen_anchor() -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load the frozen Anchor Model and Tokenizer.

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        FileNotFoundError: If the frozen model directory doesn't exist.
    """
    if not FROZEN_ANCHOR_DIR.exists():
        raise FileNotFoundError(
            f"Frozen Anchor Model not found at: {FROZEN_ANCHOR_DIR}\n"
            f"Please run  python anchor/freeze_anchor.py  first."
        )

    log.info("Loading Frozen Anchor from: %s", FROZEN_ANCHOR_DIR)
    
    model = AutoModelForCausalLM.from_pretrained(str(FROZEN_ANCHOR_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(FROZEN_ANCHOR_DIR))
    
    # Ensure pad token is set (DistilGPT2 uses EOS for pad)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

# ============================================================
# VERIFICATION CHECKS
# ============================================================

def check_frozen_weights(model: PreTrainedModel) -> bool:
    """Verify that all parameters have requires_grad=False."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            log.error("Parameter %s has requires_grad=True", name)
            return False
    return True

def run_forward_pass(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizerBase, 
    prompt: str
) -> bool:
    """
    Run a forward pass to inspect hidden states and logits.
    
    Args:
        model: The frozen model.
        tokenizer: The tokenizer.
        prompt: Text to pass to the model.
        
    Returns:
        bool: True if forward pass succeeds and outputs have correct shapes.
    """
    log.info("Running forward pass with prompt: '%s'", prompt)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    batch_size, seq_len = input_ids.shape
    vocab_size = model.config.vocab_size
    num_layers = model.config.n_layer
    hidden_size = model.config.n_embd
    
    print(f"  Input sequence length : {seq_len} tokens")
    
    # Forward pass requesting hidden states
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
    # Check logits
    logits = outputs.logits
    expected_logits_shape = (batch_size, seq_len, vocab_size)
    logits_ok = logits.shape == expected_logits_shape
    
    print(f"  Logits shape          : {tuple(logits.shape)}  (Expected: {expected_logits_shape})")
    print(f"  Logits Check          : {'[PASS]' if logits_ok else '[FAIL]'}")
    
    # Check hidden states
    # Note: hidden_states is a tuple of (embedding_output, + output of each layer)
    # So length should be num_layers + 1
    hidden_states = outputs.hidden_states
    expected_num_states = num_layers + 1
    num_states_ok = len(hidden_states) == expected_num_states
    
    print(f"  Hidden states layers  : {len(hidden_states)}  (Expected: {expected_num_states})")
    print(f"  Hidden layers Check   : {'[PASS]' if num_states_ok else '[FAIL]'}")
    
    # Check shape of last hidden state
    last_hidden = hidden_states[-1]
    expected_hidden_shape = (batch_size, seq_len, hidden_size)
    hidden_shape_ok = last_hidden.shape == expected_hidden_shape
    
    print(f"  Final hidden shape    : {tuple(last_hidden.shape)}  (Expected: {expected_hidden_shape})")
    print(f"  Hidden shape Check    : {'[PASS]' if hidden_shape_ok else '[FAIL]'}")
    
    # Get top prediction for the last token
    next_token_logits = logits[0, -1, :]
    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
    top_prob, top_idx = torch.max(probs, dim=-1)
    predicted_token = tokenizer.decode([top_idx.item()])
    
    print()
    print(f"  Top prediction for next token: '{predicted_token}' (prob: {top_prob:.4f})")
    
    return logits_ok and num_states_ok and hidden_shape_ok

def generate_text(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizerBase, 
    prompt: str,
    max_new_tokens: int = 30
) -> None:
    """
    Generate text using the model to verify it produces coherent output.
    """
    log.info("Generating text (max_new_tokens=%d)...", max_new_tokens)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
        
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print("  Generated Text:")
    print(f"  {'-' * 56}")
    
    # safe encode for CP1252 terminals
    enc = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
    safe = generated_text.encode(enc, errors="replace").decode(enc, errors="replace")
    
    # Print with basic wrapping for readability
    import textwrap
    wrapped = textwrap.fill(safe, width=56, initial_indent="  ", subsequent_indent="  ")
    print(wrapped)
    print(f"  {'-' * 56}")

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
    Execute the Anchor Model verification pipeline.
    """
    _section("Final Anchor Model Verification")

    # ── Step 1: Load ───────────────────────────────────────────────────
    try:
        model, tokenizer = load_frozen_anchor()
    except FileNotFoundError as e:
        log.error(e)
        sys.exit(1)
        
    model.eval()  # Ensure evaluation mode

    # ── Step 2: Check Weights ──────────────────────────────────────────
    _section("1. Verifying Weights are Frozen")
    is_frozen = check_frozen_weights(model)
    if is_frozen:
        print("  [PASS] All parameters have requires_grad=False.")
    else:
        print("  [FAIL] Found unfrozen parameters!")

    # ── Step 3: Forward Pass ───────────────────────────────────────────
    _section("2. Verifying Forward Pass (Logits & Hidden States)")
    forward_ok = run_forward_pass(model, tokenizer, TEST_PROMPT)

    # ── Step 4: Generation ─────────────────────────────────────────────
    _section("3. Verifying Generation Capability")
    generate_text(model, tokenizer, TEST_PROMPT)

    # ── Final Report ───────────────────────────────────────────────────
    _section("Verification Summary")
    
    all_ok = is_frozen and forward_ok
    if all_ok:
        print("  [OK] Anchor Model verification COMPLETED SUCCESSFULLY.")
        print("  The Anchor Model is ready for use in curriculum training.")
    else:
        print("  [!!] Anchor Model verification FAILED. See above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
