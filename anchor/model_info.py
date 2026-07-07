"""
Module A: Step 3 - Inspect Model Architecture & Configuration

This script inspects and displays the configuration parameters of the pretrained
DistilGPT2 model. It extracts key structural hyperparameters such as hidden size,
number of layers, attention heads, maximum positions, activation function, and
dropout rates, verifying they match expected theoretical values.

Deep Learning Concepts:
- Hidden Size (d_model): The dimensionality of the token embeddings and the hidden states
  passed between transformer layers. For DistilGPT2, this is 768.
- Attention Heads: The number of parallel self-attention operations. Multi-head attention
  allows the model to jointly attend to information from different representation subspaces.
- Transformer Layers: The number of stacked transformer blocks. DistilGPT2 has 6 layers
  (distilled from GPT-2's 12 layers).
- Position Embeddings: Static or learned vectors added to token embeddings to inject
  sequence order information since attention itself is permutation-invariant.
"""

import sys
import logging
from transformers import AutoConfig, GPT2Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "distilgpt2"


def display_model_info(model_name: str = MODEL_NAME) -> None:
    """Loads and displays key configuration parameters of the specified transformer model.

    Args:
        model_name (str): The model name/identifier to fetch the config for.
    """
    # Ensure stdout/stderr handles UTF-8 on Windows to avoid UnicodeEncodeError
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

    try:
        logger.info(f"Loading configuration for: {model_name}")
        # AutoConfig dynamically loads the configuration class associated with the model name
        config = AutoConfig.from_pretrained(model_name)
        
        # Verify it is indeed a GPT-2 style configuration
        if not isinstance(config, GPT2Config):
            logger.warning(f"Loaded config is of type {type(config)}, expected GPT2Config.")
            
    except Exception as e:
        logger.error(f"Failed to load configuration for model '{model_name}': {e}")
        return

    # Extract configuration attributes
    # In Hugging Face GPT-2 implementation:
    # - vocab_size: vocabulary size
    # - n_embd: hidden size (embedding dimension)
    # - n_layer: number of transformer layers
    # - n_head: number of attention heads
    # - n_positions: maximum sequence length / position embeddings
    # - activation_function: activation function in the MLP (Feed Forward) blocks
    # - resid_pdrop: dropout probability for residual connections
    vocab_size = getattr(config, "vocab_size", "N/A")
    hidden_size = getattr(config, "n_embd", "N/A")
    num_layers = getattr(config, "n_layer", "N/A")
    num_heads = getattr(config, "n_head", "N/A")
    max_positions = getattr(config, "n_positions", "N/A")
    activation_fn = getattr(config, "activation_function", "N/A")
    dropout_rate = getattr(config, "resid_pdrop", "N/A")

    print("\n" + "=" * 60)
    print("Architectural Parameters")
    print("=" * 60)
    print(f"Model Name                   : {model_name}")
    print(f"Vocabulary Size              : {vocab_size} (Expected: 50257)")
    print(f"Hidden Size                  : {hidden_size} (Expected: 768)")
    print(f"Number of Transformer Layers : {num_layers} (Expected: 6)")
    print(f"Number of Attention Heads    : {num_heads} (Expected: 12)")
    print(f"Maximum Position Embeddings  : {max_positions} (Expected: 1024)")
    print(f"Activation Function          : {activation_fn} (Expected: gelu or gelu_new)")
    print(f"Dropout Rate                 : {dropout_rate}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Complete Configuration Object")
    print("=" * 60)
    print(config)
    print("=" * 60)


if __name__ == "__main__":
    display_model_info()
