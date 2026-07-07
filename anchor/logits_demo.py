"""
Module A: Milestone 2, Step 3 - Logits Analysis

This script extracts the final logits from a DistilGPT2 forward pass and explains
what they represent, how they relate to the vocabulary, and how softmax converts
them into probabilities.

Deep Learning Concepts:
- Logits: Raw, unnormalized scores produced by the LM Head (a linear projection layer
  on top of the last hidden state). One score per vocabulary token.
- Softmax: A function that converts a vector of real numbers into a probability
  distribution that sums to 1.0. It amplifies the highest values and suppresses the rest.
- Temperature: A scaling factor applied to logits before softmax. Higher temperature →
  flatter distribution (more random). Lower → sharper distribution (more deterministic).
- Greedy Decoding: Always picks the token with the highest probability (argmax).
- Probability: softmax(logits)[i] = how likely the model believes token[i] is next.
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
SENTENCE = "Artificial Intelligence is changing the world."
TOP_K = 10  # Number of top predictions to display


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
        logger.error(f"Failed to load model '{model_name}': {e}")
        raise


def run_logits_demo(tokenizer: PreTrainedTokenizer, model: PreTrainedModel, text: str = SENTENCE, top_k: int = TOP_K) -> None:
    """Extracts last-token logits, converts to probabilities, and displays top-k predictions.

    Args:
        tokenizer (PreTrainedTokenizer): Loaded tokenizer.
        model (PreTrainedModel): Loaded model.
        text (str): Input sentence.
        top_k (int): Number of top predictions to show.
    """
    inputs = tokenizer(text, return_tensors="pt")
    input_ids: torch.Tensor = inputs["input_ids"]
    tokens: List[str] = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    vocab_size: int = model.config.vocab_size

    print("\n" + "=" * 60)
    print("Input")
    print("=" * 60)
    print(f"Sentence : {text}")
    print(f"Tokens   : {tokens}")

    # Run forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    logits: torch.Tensor = outputs.logits
    # Shape: [batch_size=1, sequence_length=8, vocab_size=50257]

    print("\n" + "=" * 60)
    print("Logits Tensor Overview")
    print("=" * 60)
    print(f"Full Logits Shape : {list(logits.shape)}")
    print(f"# Each token position generates {vocab_size} logit values.")
    print(f"# logits[0, i, :] represents the predicted next-token scores after position i.")
    print(f"# We are interested in the LAST token's logits: logits[0, -1, :]")
    print(f"# because it encodes the model's prediction of what comes AFTER the full sentence.")

    # Extract last token's logits (the final position sees all previous tokens via attention)
    last_token_logits: torch.Tensor = logits[0, -1, :]  # Shape: [50257]

    print("\n" + "=" * 60)
    print("Last Token Logits")
    print("=" * 60)
    print(f"Last Token Position : {len(tokens) - 1}  (token = '{tokens[-1]}')")
    print(f"Logits Shape        : {list(last_token_logits.shape)}")
    print(f"Logits Min          : {last_token_logits.min().item():.4f}")
    print(f"Logits Max          : {last_token_logits.max().item():.4f}")
    print(f"Logits Mean         : {last_token_logits.mean().item():.4f}")
    print("# These are raw floating point numbers. They can be negative.")
    print("# The token with the highest logit value is what the model predicts next (greedy).")

    # Convert logits to probabilities using softmax
    # Softmax formula: P(i) = exp(logits[i]) / sum(exp(logits[j]) for all j)
    probabilities: torch.Tensor = F.softmax(last_token_logits, dim=-1)  # Shape: [50257]
    assert abs(probabilities.sum().item() - 1.0) < 1e-3, "Probabilities must sum to 1.0"

    print("\n" + "=" * 60)
    print("Softmax → Probability Distribution")
    print("=" * 60)
    print(f"Sum of all probabilities : {probabilities.sum().item():.6f}")
    print(f"# After softmax, all values are in range [0, 1] and sum to exactly 1.0.")
    print(f"# The softmax 'squashes' a wide range of logit values into a probability distribution.")
    print(f"# Tokens with very high logits get a disproportionately high probability (amplification effect).")

    # Get top-k predictions
    top_k_values, top_k_indices = torch.topk(probabilities, k=top_k)

    print("\n" + "=" * 60)
    print(f"Top {top_k} Predicted Next Tokens")
    print("=" * 60)
    print(f"{'Rank':<6} {'Token':<20} {'Token ID':<12} {'Logit':>10} {'Probability':>14}")
    print("-" * 64)
    for rank, (token_id, prob) in enumerate(zip(top_k_indices.tolist(), top_k_values.tolist()), start=1):
        token_text = tokenizer.decode([token_id])
        logit_val = last_token_logits[token_id].item()
        print(f"{rank:<6} {repr(token_text):<20} {token_id:<12} {logit_val:>10.4f} {prob:>14.6f}")

    greedy_token_id = top_k_indices[0].item()
    greedy_token = tokenizer.decode([greedy_token_id])

    print("\n" + "=" * 60)
    print("Greedy Next Token Prediction")
    print("=" * 60)
    print(f"Predicted Token ID : {greedy_token_id}")
    print(f"Predicted Token    : {repr(greedy_token)}")
    print(f"Probability        : {top_k_values[0].item():.6f}")

    print("\n" + "=" * 60)
    print("Explanation: Logits vs Softmax vs Probability vs Prediction")
    print("=" * 60)
    print(
        "• Logits:\n"
        "  Raw, unnormalized scores from the LM Head linear layer.\n"
        "  Can be any real number (positive or negative).\n"
        "  Not a probability — they don't sum to 1.\n"
        "  Used as input to softmax.\n"
        "\n"
        "• Softmax:\n"
        "  A mathematical function that converts logits into probabilities.\n"
        "  Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)\n"
        "  Preserves relative ordering: the token with the highest logit\n"
        "  still gets the highest probability after softmax.\n"
        "\n"
        "• Probability:\n"
        "  The output of softmax. A value between 0 and 1.\n"
        "  Represents the model's confidence that a given token is the next token.\n"
        "  All probabilities sum to exactly 1.0 (they form a distribution).\n"
        "\n"
        "• Prediction (Greedy Decoding):\n"
        "  Simply select argmax(probabilities) — the token with the highest probability.\n"
        "  This is the simplest decoding strategy.\n"
        "  Other strategies (beam search, top-p sampling) explore more possibilities."
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
        run_logits_demo(tokenizer, model, SENTENCE)
    except Exception as e:
        logger.critical(f"Logits demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
