"""
Module A: Milestone 2, Step 1 - Forward Pass Demonstration

This script demonstrates one complete forward pass through the DistilGPT2 model.
It explains exactly what happens when input_ids are passed through model() and
what the output object contains.

Deep Learning Concepts:
- Forward Pass: The computation of passing input data through the network from
  input to output without updating any weights.
- Logits: Raw, unnormalized scores output by the final linear layer (the LM head)
  over the vocabulary. Shape: (batch_size, sequence_length, vocab_size).
- Hidden States: The intermediate vector representations at every layer of the transformer.
  Each state captures a progressively deeper semantic abstraction of the input.
- BaseModelOutputWithPast: The structured output dataclass returned by Hugging Face
  causal LM models, containing logits, past_key_values, and optionally hidden states/attentions.
"""

import sys
import logging
from typing import Tuple
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
    modeling_outputs,
)

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


def load_model_and_tokenizer(model_name: str = MODEL_NAME) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Loads pretrained tokenizer and causal LM from Hugging Face.

    Args:
        model_name (str): Hugging Face model identifier.

    Returns:
        Tuple[PreTrainedTokenizer, PreTrainedModel]: Tokenizer and model.

    Raises:
        Exception: If loading fails.
    """
    try:
        logger.info(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # Set model to eval mode: disables dropout layers for deterministic behavior
        model.eval()
        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to load model/tokenizer '{model_name}': {e}")
        raise


def run_forward_pass(tokenizer: PreTrainedTokenizer, model: PreTrainedModel, text: str = SENTENCE) -> None:
    """Runs a single forward pass through DistilGPT2 and inspects the output structure.

    Args:
        tokenizer (PreTrainedTokenizer): The loaded tokenizer.
        model (PreTrainedModel): The loaded causal language model.
        text (str): Input sentence.
    """
    # Tokenize the sentence and build PyTorch input tensors
    inputs = tokenizer(text, return_tensors="pt")
    input_ids: torch.Tensor = inputs["input_ids"]

    print("\n" + "=" * 60)
    print("Input Sentence and Tokenization")
    print("=" * 60)
    print(f"Sentence  : {text}")
    print(f"input_ids : {input_ids}")
    print(f"Shape     : {list(input_ids.shape)}")
    print("# Shape breakdown: [batch_size=1, sequence_length=8]")
    print("# The model processes every token in the sequence simultaneously (not one by one).")

    print("\n" + "=" * 60)
    print("Forward Pass: Calling model(input_ids)")
    print("=" * 60)
    print("# model() triggers the full forward pass:")
    print("#   1. Embedding lookup: token_ids -> dense vectors")
    print("#   2. Position encoding: position info added to embeddings")
    print("#   3. 6 Transformer blocks: each applies Self-Attention + MLP")
    print("#   4. LM Head (linear layer): projects final hidden state -> vocabulary logits")

    # No gradient computation needed for inference
    with torch.no_grad():
        outputs = model(**inputs)

    print("\n" + "=" * 60)
    print("Output Object Type")
    print("=" * 60)
    print(f"Type : {type(outputs)}")
    print("# CausalLMOutputWithCrossAttentions is a dataclass from Hugging Face.")
    print("# It is a named container (like an OrderedDict) holding all outputs.")
    print("# Using a dataclass allows accessing values by name (outputs.logits) or by key ('logits').")

    print("\n" + "=" * 60)
    print("Available Keys in Output")
    print("=" * 60)
    for key, value in outputs.items():
        if value is not None:
            shape_str = str(list(value.shape)) if isinstance(value, torch.Tensor) else f"tuple of {len(value)} elements"
            print(f"  {key:30s}: {shape_str}")
        else:
            print(f"  {key:30s}: None (not requested)")
    print("# Keys that are None were not requested (e.g., hidden_states requires output_hidden_states=True).")

    print("\n" + "=" * 60)
    print("Logits Shape")
    print("=" * 60)
    logits: torch.Tensor = outputs.logits
    print(f"Shape : {list(logits.shape)}")
    print(f"# Logits shape breakdown:")
    print(f"#   [batch_size=1, sequence_length=8, vocab_size=50257]")
    print(f"# At every position in the sequence, the model outputs one score per vocabulary token.")
    print(f"# The score at position i represents 'how likely is token[i+1] given tokens[0..i]'")
    print(f"# These raw scores are called logits — they are NOT probabilities yet.")
    print(f"# To convert to probabilities: apply softmax over the vocab_size dimension.")

    print("\n" + "=" * 60)
    print("Past Key Values (KV-Cache) Info")
    print("=" * 60)
    pkv = outputs.past_key_values
    if pkv is not None:
        # In Transformers >= 4.38, past_key_values is a DynamicCache object (not a plain tuple).
        # DynamicCache exposes .key_cache and .value_cache lists, and len() returns num layers.
        num_layers = len(pkv)
        print(f"KV-Cache Type          : {type(pkv).__name__}")
        print(f"Num Layers in KV-Cache : {num_layers}")
        # Access the key cache for the first layer to show tensor shape
        try:
            # DynamicCache stores tensors in .key_cache / .value_cache (list indexed by layer)
            if hasattr(pkv, 'key_cache') and len(pkv.key_cache) > 0:
                print(f"Shape per layer (keys) : {list(pkv.key_cache[0].shape)}")
            elif hasattr(pkv, '__getitem__'):
                # Fallback: older tuple-of-tuples format
                print(f"Shape per layer (keys) : {list(pkv[0][0].shape)}")
        except Exception:
            print("(Shape introspection not available for this cache type)")
        print("# past_key_values stores the key/value projections from self-attention.")
        print("# This is the KV-Cache mechanism — precomputed attention states are reused")
        print("# at each generation step to avoid redundant computation over the full context.")
    else:
        print("past_key_values : None")

    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    print(
        "  Sentence\n"
        "       ↓\n"
        "  Tokenizer          [8001, 9542, 9345, 318, 5609, 262, 995, 13]\n"
        "       ↓\n"
        "  Embedding Layer    [1, 8, 768] dense vectors\n"
        "       ↓\n"
        "  6 Transformer Blocks (Self-Attention + MLP)\n"
        "       ↓\n"
        "  LM Head (Linear)   [1, 8, 50257] logits\n"
        "       ↓\n"
        "  Softmax → Next Token Probability Distribution"
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
        run_forward_pass(tokenizer, model, SENTENCE)
    except Exception as e:
        logger.critical(f"Forward pass demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
