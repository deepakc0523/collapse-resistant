"""
Module A: Milestone 2, Step 4 - Next-Token Prediction Demo

This script demonstrates autoregressive text generation using DistilGPT2.
It shows how the model predicts one token at a time, and illustrates the
full pipeline from an input prompt through to a completed generated sequence.

Deep Learning Concepts:
- Autoregressive Generation: At each step, the model generates one new token by
  conditioning on all previously generated tokens. The output of step t becomes
  the input context for step t+1.
- Greedy Decoding: At each step, select the single token with the highest probability.
  Deterministic, fast, but can produce repetitive or suboptimal sequences.
- Greedy vs Sampling: Greedy always picks argmax. Sampling (do_sample=True) draws
  from the probability distribution, producing more diverse outputs.
- model.generate(): The Hugging Face API for text generation. Internally it runs
  the forward pass iteratively, appending the best next token each time.
"""

import sys
import logging
from typing import List, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "distilgpt2"
PROMPT = "Artificial Intelligence is changing the world."
MAX_NEW_TOKENS = 20


def load_model_and_tokenizer(model_name: str = MODEL_NAME) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Loads pretrained tokenizer and model.

    Args:
        model_name (str): Hugging Face model identifier.

    Returns:
        Tuple[PreTrainedTokenizer, PreTrainedModel]: Tokenizer and model.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to load: {e}")
        raise


def run_step_by_step_prediction(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    prompt: str = PROMPT,
    num_steps: int = 5
) -> None:
    """Demonstrates greedy next-token prediction step by step for a limited number of steps.

    Args:
        tokenizer (PreTrainedTokenizer): Loaded tokenizer.
        model (PreTrainedModel): Loaded model.
        prompt (str): Initial text prompt.
        num_steps (int): Number of token generation steps to demonstrate.
    """
    print("\n" + "=" * 60)
    print(f"Step-by-Step Greedy Decoding (first {num_steps} tokens)")
    print("=" * 60)
    print("# At each step:")
    print("#   1. Tokenize the current context")
    print("#   2. Run a forward pass")
    print("#   3. Extract last-position logits")
    print("#   4. Apply softmax → probability distribution")
    print("#   5. Pick argmax → next token (greedy)")
    print("#   6. Append to context → repeat")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids: torch.Tensor = inputs["input_ids"]

    current_ids = input_ids.clone()

    for step in range(num_steps):
        with torch.no_grad():
            outputs = model(current_ids)

        # Logits at last position shape: [vocab_size]
        last_logits = outputs.logits[0, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        next_token_id = torch.argmax(probs).item()
        next_token_str = tokenizer.decode([next_token_id])
        next_token_prob = probs[next_token_id].item()

        print(f"\n  Step {step + 1}")
        print(f"    Context length : {current_ids.shape[1]} tokens")
        print(f"    Next Token ID  : {next_token_id}")
        print(f"    Next Token     : {repr(next_token_str)}")
        print(f"    Probability    : {next_token_prob:.6f}")

        # Append the predicted token to the current sequence
        next_token_tensor = torch.tensor([[next_token_id]])
        current_ids = torch.cat([current_ids, next_token_tensor], dim=1)

    print(f"\n  Context after {num_steps} greedy steps:")
    print(f"  {repr(tokenizer.decode(current_ids[0].tolist()))}")


def run_prediction_demo(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    prompt: str = PROMPT,
    max_new_tokens: int = MAX_NEW_TOKENS
) -> None:
    """Uses model.generate() to generate text and analyzes the results.

    Args:
        tokenizer (PreTrainedTokenizer): Loaded tokenizer.
        model (PreTrainedModel): Loaded model.
        prompt (str): Input text.
        max_new_tokens (int): Number of tokens to generate.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids: torch.Tensor = inputs["input_ids"]
    prompt_token_count: int = input_ids.shape[1]

    print("\n" + "=" * 60)
    print("Input Prompt")
    print("=" * 60)
    print(f"Prompt : {prompt}")
    print(f"Prompt Token Count : {prompt_token_count}")

    print("\n" + "=" * 60)
    print(f"Generating {max_new_tokens} New Tokens (Greedy, do_sample=False)")
    print("=" * 60)
    print("# model.generate() runs the forward pass iteratively.")
    print("# do_sample=False  → greedy decoding (deterministic, picks argmax each step)")
    print("# do_sample=True   → random sampling from the probability distribution (stochastic)")

    with torch.no_grad():
        generated_ids: torch.Tensor = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # Greedy decoding
            temperature=1.0,           # No temperature scaling (default)
            pad_token_id=tokenizer.eos_token_id,  # Use eos_token as pad to avoid warning
        )

    # generated_ids includes the original prompt tokens + newly generated tokens
    all_token_ids: List[int] = generated_ids[0].tolist()
    new_token_ids: List[int] = all_token_ids[prompt_token_count:]

    # Decode full output and individual tokens
    full_text: str = tokenizer.decode(all_token_ids, skip_special_tokens=True)
    new_tokens: List[str] = [tokenizer.decode([tid]) for tid in new_token_ids]

    print("\n" + "=" * 60)
    print("Generated Output")
    print("=" * 60)
    print(f"Full Text:\n  {full_text}")

    print("\n" + "=" * 60)
    print("Generated Token IDs")
    print("=" * 60)
    print(new_token_ids)

    print("\n" + "=" * 60)
    print("Generated Tokens (Decoded)")
    print("=" * 60)
    for i, (tid, tok) in enumerate(zip(new_token_ids, new_tokens), start=1):
        print(f"  Step {i:>2}: ID={tid:<7} Token={repr(tok)}")

    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    print(
        "  Input Sentence\n"
        "       ↓\n"
        "  Tokenizer → input_ids\n"
        "       ↓\n"
        "  Embedding Layer → [seq_len, 768] dense vectors\n"
        "       ↓\n"
        "  6 Transformer Blocks → contextual hidden states\n"
        "       ↓\n"
        "  LM Head → logits [vocab_size=50257] at last position\n"
        "       ↓\n"
        "  Softmax → Probability Distribution\n"
        "       ↓\n"
        "  Argmax (greedy) → Next Token ID\n"
        "       ↓\n"
        "  Append to context → Repeat for next step"
    )

    print("\n" + "=" * 60)
    print("Explanation: Greedy Decoding and Autoregressive Generation")
    print("=" * 60)
    print(
        "• Greedy Decoding:\n"
        "  At every step, the token with the highest probability is selected.\n"
        "  It is deterministic — given the same input and model, output is always identical.\n"
        "  It is fast (no beam search, no sampling overhead).\n"
        "  Disadvantage: can fall into repetitive loops or miss globally better sequences.\n"
        "\n"
        "• Autoregressive Generation:\n"
        "  The model is trained to predict one token at a time, left to right.\n"
        "  During generation, the predicted token is appended to the input,\n"
        "  and the entire expanded sequence is passed back to the model for the next step.\n"
        "  This is why it's called 'autoregressive' — it uses its own past outputs as inputs.\n"
        "\n"
        "• Why prediction is simply argmax:\n"
        "  Logits → softmax → probabilities → argmax gives the greedy next token.\n"
        "  This is deterministic and does not require sampling from any distribution.\n"
        "  More sophisticated decoding (beam search, top-p/top-k sampling) explore\n"
        "  alternate paths through the probability space for higher quality generations."
    )
    print("=" * 60)


def main() -> None:
    """Main execution entry point."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

    try:
        tokenizer, model = load_model_and_tokenizer(MODEL_NAME)

        # Part 1: Manual step-by-step greedy prediction for 5 steps
        run_step_by_step_prediction(tokenizer, model, PROMPT, num_steps=5)

        # Part 2: Use model.generate() for full generation of MAX_NEW_TOKENS
        run_prediction_demo(tokenizer, model, PROMPT, max_new_tokens=MAX_NEW_TOKENS)

    except Exception as e:
        logger.critical(f"Prediction demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
