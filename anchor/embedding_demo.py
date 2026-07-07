"""
Module A: Step 4 - Embedding Layer Analysis

This script demonstrates and analyzes the Embedding Layer of the DistilGPT2 model.
It explains and illustrates the pipeline of converting text tokens to high-dimensional,
dense continuous vectors, and calculates relevant statistics and parameter sizes.

Deep Learning Concepts:
- Word Embedding: A dense vector representation of a word or subword token in a continuous vector space.
- nn.Embedding: A lookup table that maps discrete token IDs to dense vectors of a fixed size.
- Continuous Vector Space: Allows semantic relationships to be captured geometrically (e.g., via cosine similarity).
"""

import sys
import logging
from typing import Tuple, List
import torch
import torch.nn as nn
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


def load_model_and_tokenizer(model_name: str = MODEL_NAME) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Loads the pretrained tokenizer and model from Hugging Face.

    Args:
        model_name (str): Hugging Face model identifier.

    Returns:
        Tuple[PreTrainedTokenizer, PreTrainedModel]: Tokenizer and Model objects.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading model or tokenizer for '{model_name}': {e}")
        raise


def run_embedding_demo(tokenizer: PreTrainedTokenizer, model: PreTrainedModel, text: str = SENTENCE) -> None:
    """Processes a text sentence through the model's embedding layer,

    computes architectural statistics, performs validations, and prints
    detailed explanations.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to process text.
        model (PreTrainedModel): Transformer model containing embedding layer.
        text (str): Input sentence.
    """
    # 1. Tokenization pipeline
    tokens: List[str] = tokenizer.tokenize(text)
    token_ids: List[int] = tokenizer.convert_tokens_to_ids(tokens)
    
    # Format as PyTorch batch inputs: batch_size = 1
    inputs = tokenizer(text, return_tensors="pt")
    input_ids: torch.Tensor = inputs["input_ids"]
    batch_size, sequence_length = input_ids.shape

    # 2. Extract embedding layer directly from the model
    # get_input_embeddings() returns the token embedding module (typically nn.Embedding)
    embedding_layer: nn.Module = model.get_input_embeddings()
    if not isinstance(embedding_layer, nn.Embedding):
        raise TypeError(f"Expected nn.Embedding, but got {type(embedding_layer)}")

    # Retrieve weight matrix parameters
    embedding_matrix: torch.Tensor = embedding_layer.weight
    vocab_size, hidden_size = embedding_matrix.shape
    total_embedding_params = embedding_matrix.numel()

    # 3. Pass input IDs through the embedding layer
    # This retrieves corresponding dense vectors from the embedding lookup table
    with torch.no_grad():
        embedding_tensor: torch.Tensor = embedding_layer(input_ids)

    # 4. Verifications
    # Retrieve hidden size and vocab size from model configuration to check consistency
    config_hidden_size = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.n_embd
    config_vocab_size = model.config.vocab_size

    if hidden_size != config_hidden_size:
        raise ValueError(
            f"Verification Failed: Embedding dimension ({hidden_size}) "
            f"does not match model config hidden size ({config_hidden_size})"
        )
    if vocab_size != config_vocab_size:
        raise ValueError(
            f"Verification Failed: Embedding matrix vocab size ({vocab_size}) "
            f"does not match model config vocab size ({config_vocab_size})"
        )
    expected_tensor_shape = (batch_size, sequence_length, hidden_size)
    if embedding_tensor.shape != expected_tensor_shape:
        raise ValueError(
            f"Verification Failed: Embedding tensor shape {embedding_tensor.shape} "
            f"does not match expected shape {expected_tensor_shape}"
        )

    # 5. Extract statistics
    # Stats of the actual embedding vectors computed for our sentence
    min_val = embedding_tensor.min().item()
    max_val = embedding_tensor.max().item()
    mean_val = embedding_tensor.mean().item()
    std_val = embedding_tensor.std().item()

    # Extract first token embedding (first 20 dimensions)
    first_token_emb_20 = embedding_tensor[0, 0, :20].tolist()

    # Display Outputs in requested order
    print("\n" + "=" * 60)
    print("Sentence")
    print("=" * 60)
    print(text)

    print("\n" + "=" * 60)
    print("Tokens")
    print("=" * 60)
    print(tokens)

    print("\n" + "=" * 60)
    print("Token IDs")
    print("=" * 60)
    print(token_ids)

    print("\n" + "=" * 60)
    print("Embedding Matrix Information")
    print("=" * 60)
    print(f"Vocabulary Size     : {vocab_size}")
    print(f"Embedding Dimension : {hidden_size}")
    print(f"Embedding Matrix    : {vocab_size} x {hidden_size}")
    print(f"Embedding Parameters: {total_embedding_params:,}")
    print(
        "# Note: The embedding matrix is one of the largest parameter matrices in GPT.\n"
        "# Because vocabulary size is massive (~50k tokens) and each token requires a dense\n"
        "# vector representation, the input embedding matrix maps discrete indices to continuous\n"
        "# vector space. For DistilGPT2, this layer alone constitutes roughly 38.6M parameters\n"
        "# (50,257 * 768), which is a substantial percentage (~47%) of the total model parameter count."
    )

    print("\n" + "=" * 60)
    print("Embedding Tensor")
    print("=" * 60)
    print(f"Tensor Shape : {list(embedding_tensor.shape)}")
    print(
        f"# Output format: (batch_size, sequence_length, hidden_size) = ({batch_size}, {sequence_length}, {hidden_size})"
    )

    print("\n" + "=" * 60)
    print("First Token Embedding (First 20 Values)")
    print("=" * 60)
    print([round(v, 6) for v in first_token_emb_20])

    print("\n" + "=" * 60)
    print("Embedding Statistics")
    print("=" * 60)
    print(f"Minimum            : {min_val:.6f}")
    print(f"Maximum            : {max_val:.6f}")
    print(f"Mean               : {mean_val:.6f}")
    print(f"Standard Deviation : {std_val:.6f}")

    print("\n" + "=" * 60)
    print("Explanation")
    print("=" * 60)
    print(
        "• What an embedding is:\n"
        "  An embedding is a dense, low-dimensional, continuous-valued vector representation of discrete\n"
        "  categorical items (like words or subwords). It maps discrete, non-differentiable IDs into a \n"
        "  vector space where geometric operations (like distance and direction) reflect semantic relationships.\n"
        "\n"
        "• Why embeddings are learned:\n"
        "  Initial embeddings are randomized. During training, backpropagation adjusts these parameters\n"
        "  so that tokens appearing in similar contexts (e.g., 'cat' and 'dog', or 'Intelligence' and 'AI')\n"
        "  are pulled closer together in the continuous space, forming clusters representing semantic similarity.\n"
        "\n"
        "• Why every token becomes a 768-dimensional vector:\n"
        "  The dimension size (768) is a design choice (hyperparameter d_model). It controls the capacity\n"
        "  of the model's representation. A larger dimension allows encoding more nuanced relationships and facts,\n"
        "  but increases memory requirements and risks overfitting.\n"
        "\n"
        "• Why embeddings are the first layer of the transformer:\n"
        "  Computers cannot directly compute calculus/gradients on strings. Transformers rely on linear layers\n"
        "  and matrix multiplications. Converting discrete tokens into dense continuous vectors (embeddings) is\n"
        "  the essential interface, mapping human language to numerical tensors suitable for deep neural networks."
    )
    print("=" * 60)


def main() -> None:
    """Main execution point to load models, configure stdout, run demo, and catch errors."""
    # Ensure stdout/stderr handles UTF-8 on Windows to avoid UnicodeEncodeError
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

    try:
        tokenizer, model = load_model_and_tokenizer(MODEL_NAME)
        run_embedding_demo(tokenizer, model, SENTENCE)
    except Exception as e:
        logger.critical(f"Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
