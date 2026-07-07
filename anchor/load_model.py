"""
Module A: Step 1 - Load Pretrained Model

This script loads the pretrained DistilGPT2 model and its associated tokenizer
using the Hugging Face Transformers library. It prints key architectural metrics
such as the model name, vocabulary size, and total number of parameters to verify
successful loading.

Concepts:
- Tokenizer: Converts text into a sequence of integer token IDs mapping to a predefined vocabulary.
- Causal Language Model (CausalLM): A model trained to predict the next token given preceding context.
- DistilGPT2: A distilled version of GPT-2, containing 6 layers (instead of GPT-2's 12) while preserving much of the capability.
"""

import sys
import logging
from typing import Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel

# Configure logging to output clean, readable logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "distilgpt2"


def load_model(model_name: str = MODEL_NAME) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Loads the tokenizer and pretrained causal language model from Hugging Face.

    Args:
        model_name (str): The identifier of the model to load from Hugging Face Hub.

    Returns:
        Tuple[PreTrainedTokenizer, PreTrainedModel]: A tuple containing the loaded tokenizer
        and the model.

    Raises:
        Exception: If downloading or loading the tokenizer/model fails.
    """
    try:
        logger.info("=" * 60)
        logger.info(f"Loading Tokenizer: {model_name}...")
        logger.info("=" * 60)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer Loaded\n")

        logger.info("=" * 60)
        logger.info(f"Loading Model: {model_name}...")
        logger.info("=" * 60)
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("✓ Model Loaded\n")
        
        return tokenizer, model

    except Exception as e:
        logger.error(f"Failed to load tokenizer or model '{model_name}': {e}")
        logger.error("Please verify your internet connection and that the model name is correct.")
        raise


def main() -> None:
    """Main execution block to load the model and display standard model metadata."""
    # Ensure stdout/stderr handles UTF-8 on Windows to avoid UnicodeEncodeError for character '✓'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

    try:
        tokenizer, model = load_model(MODEL_NAME)
        
        # Calculate parameters
        # GPT-2 / DistilGPT2 parameters are split into token embeddings, position embeddings,
        # transformer layers (attention, MLP), and final layer normalization.
        total_params = model.num_parameters()
        vocab_size = len(tokenizer)  # Use len(tokenizer) to include special tokens if added

        print("=" * 60)
        print("Model Information")
        print("=" * 60)
        print(f"Model Name : {MODEL_NAME}")
        print(f"Vocabulary Size : {vocab_size}")
        print(f"Number of Parameters : {total_params}")
        print("=" * 60)
        
    except Exception as e:
        logger.critical(f"Execution terminated due to an error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()