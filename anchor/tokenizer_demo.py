"""
Module A: Step 2 - Tokenizer Demonstration

This script demonstrates how a raw text sentence is processed by the DistilGPT2 tokenizer
into tokens, token IDs, and PyTorch model inputs (input_ids and attention_mask).

Deep Learning Concepts:
- Byte-Pair Encoding (BPE): The tokenization algorithm used by GPT-2/DistilGPT2. It treats
  text at the byte level, allowing it to build a vocabulary of subword units, avoiding
  out-of-vocabulary (OOV) issues. A leading 'Ġ' character represents a space.
- Token IDs: Numeric representations of tokens mapping directly to the model's embedding matrix.
- input_ids: The tensor containing the vocabulary indices of the input tokens.
- attention_mask: Binary tensor (0 or 1) indicating to the model's self-attention mechanism
  which tokens are padding tokens (0) and which are actual content tokens (1).
"""

import sys
import logging
from typing import List, Dict
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "distilgpt2"
DEFAULT_SENTENCE = "Artificial Intelligence is changing the world."


def run_tokenizer_demo(sentence: str = DEFAULT_SENTENCE, model_name: str = MODEL_NAME) -> None:
    """Demonstrates tokenization, ID mapping, and input representation step-by-step.

    Args:
        sentence (str): The raw text sentence to tokenize.
        model_name (str): The name of the pretrained tokenizer to load.
    """
    # Ensure stdout/stderr handles UTF-8 on Windows to avoid UnicodeEncodeError for character 'Ġ'
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

    try:
        logger.info(f"Loading tokenizer for demo: {model_name}")
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Failed to load tokenizer '{model_name}': {e}")
        return

    print("=" * 60)
    print("Original Sentence")
    print("=" * 60)
    print(sentence)
    print("\n" + "=" * 60)
    print("Tokens")
    print("=" * 60)
    
    # 1. Tokenize: Convert string to list of subword tokens
    # Note: In GPT-2's BPE, spaces are encoded as part of the token (prefixed with 'Ġ')
    tokens: List[str] = tokenizer.tokenize(sentence)
    print(tokens)
    print("# Explanation: Raw text is split into subword tokens using Byte-Pair Encoding (BPE).")
    print("# The 'Ġ' symbol represents a space prefix, allowing the model to distinguish spaces without explicit delimiters.")
    print("\n" + "=" * 60)
    print("Token IDs")
    print("=" * 60)
    
    # 2. Token IDs: Map each string token to its corresponding vocabulary index
    token_ids: List[int] = tokenizer.convert_tokens_to_ids(tokens)
    print(token_ids)
    print("# Explanation: Each token is mapped to a unique integer index in the vocabulary (size 50,257).")
    print("\n" + "=" * 60)
    print("input_ids")
    print("=" * 60)
    
    # 3. Model Inputs (input_ids and attention_mask)
    # The tokenizer call wraps the text directly into PyTorch tensors, returning a dictionary
    encoding = tokenizer(sentence, return_tensors="pt")
    input_ids: torch.Tensor = encoding["input_ids"]
    attention_mask: torch.Tensor = encoding["attention_mask"]
    
    print(input_ids)
    print("# Explanation: 'input_ids' is the 2D PyTorch tensor (batch_size=1, sequence_length=6)")
    print("# containing the vocabulary indices. This tensor is passed to the embedding layer.")
    print("\n" + "=" * 60)
    print("attention_mask")
    print("=" * 60)
    
    print(attention_mask)
    print("# Explanation: 'attention_mask' indicates which tokens are padding (0) and which are content (1).")
    print("# Since there is no padding here, all values are 1, indicating the model should attend to all tokens.")
    print("=" * 60)


if __name__ == "__main__":
    run_tokenizer_demo()