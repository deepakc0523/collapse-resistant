"""
Module A: Milestone 2, Step 2 - Hidden States Analysis

This script extracts and analyzes the hidden states from every transformer layer
inside DistilGPT2. It demonstrates how token representations evolve from raw
embeddings (Layer 0) to high-level semantic encodings (Layer 6).

Deep Learning Concepts:
- Hidden States: The intermediate tensor representations produced at the output of each
  transformer layer. Each hidden state for a token reflects the meaning of that token in
  the context of its surrounding tokens, as understood by that layer.
- Contextualization: Unlike static embeddings, hidden states are contextual — the same
  token can produce different hidden states depending on the tokens around it.
- Representational Depth: Lower layers tend to capture syntactic/positional features,
  while deeper layers encode more abstract, semantic, and task-specific representations.
"""

import sys
import logging
from typing import List, Tuple
import torch
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
    """Loads pretrained tokenizer and model from Hugging Face.

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


def layer_label(layer_idx: int, num_hidden_layers: int) -> str:
    """Returns a descriptive label for each hidden state layer index.

    Args:
        layer_idx (int): The layer index (0-indexed).
        num_hidden_layers (int): Total number of transformer blocks.

    Returns:
        str: Human-readable label.
    """
    if layer_idx == 0:
        return "Layer 0 — Initial Token Embeddings (before any Transformer processing)"
    elif layer_idx == num_hidden_layers:
        return f"Layer {layer_idx} — Final Transformer Layer (most abstract semantic representation)"
    else:
        return f"Layer {layer_idx} — Transformer Block {layer_idx} output"


def compute_layer_stats(hidden_state: torch.Tensor) -> Tuple[float, float, float, float]:
    """Computes basic statistics over a single hidden state tensor.

    Args:
        hidden_state (torch.Tensor): Shape [batch_size, seq_len, hidden_size].

    Returns:
        Tuple[float, float, float, float]: mean, std, min, max.
    """
    return (
        hidden_state.mean().item(),
        hidden_state.std().item(),
        hidden_state.min().item(),
        hidden_state.max().item(),
    )


def run_hidden_states_demo(tokenizer: PreTrainedTokenizer, model: PreTrainedModel, text: str = SENTENCE) -> None:
    """Runs a forward pass with output_hidden_states=True and analyzes all layers.

    Args:
        tokenizer (PreTrainedTokenizer): Loaded tokenizer.
        model (PreTrainedModel): Loaded model.
        text (str): Input sentence.
    """
    inputs = tokenizer(text, return_tensors="pt")
    input_ids: torch.Tensor = inputs["input_ids"]
    tokens: List[str] = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    num_hidden_layers: int = model.config.n_layer

    print("\n" + "=" * 60)
    print("Input")
    print("=" * 60)
    print(f"Sentence       : {text}")
    print(f"Tokens         : {tokens}")
    print(f"Token IDs      : {input_ids[0].tolist()}")
    print(f"Sequence Length: {input_ids.shape[1]}")

    print("\n" + "=" * 60)
    print("Requesting All Hidden States")
    print("=" * 60)
    print("# output_hidden_states=True tells the model to return hidden states from ALL layers,")
    print("# not just the final layer. Without this flag, intermediate representations are discarded.")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states: Tuple[torch.Tensor, ...] = outputs.hidden_states

    print("\n" + "=" * 60)
    print("Hidden States Summary")
    print("=" * 60)
    print(f"Number of hidden state tensors returned : {len(hidden_states)}")
    print(f"# DistilGPT2 has {num_hidden_layers} transformer blocks.")
    print(f"# Hugging Face returns {num_hidden_layers + 1} hidden states:")
    print(f"#   - 1 initial embedding output (Layer 0)")
    print(f"#   - {num_hidden_layers} outputs from each Transformer block (Layers 1–{num_hidden_layers})")

    print("\n" + "=" * 60)
    print("Per-Layer Shape and Statistics")
    print("=" * 60)
    print(f"{'Layer':<10} {'Shape':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 72)

    for i, hs in enumerate(hidden_states):
        mean, std, mn, mx = compute_layer_stats(hs)
        shape_str = str(list(hs.shape))
        print(f"{i:<10} {shape_str:<20} {mean:>10.5f} {std:>10.5f} {mn:>10.5f} {mx:>10.5f}")

    print("\n" + "=" * 60)
    print("Per-Layer Detailed View")
    print("=" * 60)
    for i, hs in enumerate(hidden_states):
        label = layer_label(i, num_hidden_layers)
        mean, std, mn, mx = compute_layer_stats(hs)
        print(f"\n--- {label} ---")
        print(f"  Shape     : {list(hs.shape)}")
        print(f"  Mean      : {mean:.6f}")
        print(f"  Std Dev   : {std:.6f}")
        print(f"  Min       : {mn:.6f}")
        print(f"  Max       : {mx:.6f}")

    print("\n" + "=" * 60)
    print("Explanation: Why Hidden States Become More Semantic Deeper In The Network")
    print("=" * 60)
    print(
        "• Layer 0 (Embeddings):\n"
        "  The initial hidden state is simply the sum of token embeddings and position embeddings.\n"
        "  Tokens are represented independently, without any contextual awareness.\n"
        "  A word like 'bank' at this layer has the same vector regardless of context.\n"
        "\n"
        "• Layers 1–3 (Early Transformer Blocks):\n"
        "  Self-attention begins building local syntactic structure.\n"
        "  Tokens start 'communicating' with their neighbors via attention weights.\n"
        "  The model learns part-of-speech information, phrase boundaries, and basic dependencies.\n"
        "\n"
        "• Layers 4–6 (Later Transformer Blocks):\n"
        "  Representations become increasingly abstract and globally contextual.\n"
        "  'Bank' near 'river' now has a different hidden state than 'bank' near 'finance'.\n"
        "  The final hidden state (Layer 6) is used to predict the next token via the LM Head.\n"
        "\n"
        "• This progressive abstraction is the core mechanism enabling LLMs to understand language.\n"
        "  It is also why representation collapse (when hidden states become degenerate/rank-deficient)\n"
        "  is so damaging — it collapses this rich hierarchical representation into a subspace."
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
        run_hidden_states_demo(tokenizer, model, SENTENCE)
    except Exception as e:
        logger.critical(f"Hidden states demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
