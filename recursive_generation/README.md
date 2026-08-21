# Recursive Generation Module

The **Recursive Generation** module is the Generation-2 data synthesis engine in the **Collapse-Resistant Recursive Language Model Training Framework**. It loads the trained **Generation-1 Student model** (`checkpoints/student_model/best/`) and the canonical **human anchor prefix dataset** (`data/processed/clean_wikitext.txt`) to generate the Generation-2 synthetic dataset.

This module is designed to run efficiently on a **Tesla T4 GPU in Google Colab** with full AMP support, batched generation, streaming JSONL writes, and automatic resume-after-interruption.

---

## 1. Scientific Explanation

### Autoregressive Generation

The student model is a **causal language model (CLM)**. At each decoding step, it predicts the probability distribution over the full vocabulary given the prefix of tokens seen so far:

$$P(x_{t} \mid x_1, x_2, \ldots, x_{t-1}) = \text{softmax}(\text{LM\_Head}(h_t))$$

where $h_t$ is the final hidden state at position $t$ from the transformer stack.

### Sampling Strategies

**Temperature** ($T$) sharpens or flattens the next-token distribution:
$$P'(x_t) = \text{softmax}(\log P(x_t) / T)$$

- $T \to 0$: Greedy (deterministic), $T = 1$: Unmodified, $T > 1$: Flatter (more random).

**Top-K Sampling** restricts the candidate set to the top $K$ vocabulary items per step. **Top-P (Nucleus) Sampling** dynamically selects the minimum vocabulary subset whose cumulative probability exceeds $p$.

**Repetition Penalty** discounts log-probabilities of previously generated tokens to suppress repetitive loops.

### Why Human Prefixes Are Reused

The canonical human-authored prefixes from `clean_wikitext.txt` are deliberately reused across all generations. This ensures:
1. **Distributional Anchoring**: Each generation receives the same distributional starting conditions.
2. **Comparative Measurement**: Probe, Ensemble, and SCRS modules can directly compare outputs across generations on identical inputs.
3. **Recursive Integrity**: Generational drift is measured relative to the same anchor context, not drifting prompt distributions.

### Why the Student Produces Generation-2

The Generation-1 student learned an approximation of the anchor model's output distribution. By prompting it with the same human prefixes, we collect its probability-weighted completions — which will exhibit varying degrees of **representation drift** from the anchor ground truth. Generation-2 data then becomes the input to the next SCRS evaluation cycle.

---

## 2. Expected File Structure

```
recursive_generation/
├── __init__.py
├── generation_config.py
├── utils.py
├── model_loader.py
├── prefix_loader.py
├── resume_manager.py
├── generator.py
├── metadata_writer.py
├── visualization.py
├── verify_recursive_generation.py
├── run_recursive_generation.py
└── README.md
```

---

## 3. Output Specification (`recursive_generation_out/generation_2/`)

### `generation_2_synthetic.jsonl` — Record Schema
```json
{
  "prompt": "The 2011 census recorded ...",
  "generated_continuation": "a population of 12,000 ...",
  "full_text": "The 2011 census recorded a population of 12,000 ...",
  "generation": 2,
  "parent_student": "generation_1",
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.3,
  "max_new_tokens": 128,
  "seed": 42,
  "_prompt_index": 3712
}
```

### `generation_metadata.json` — Key Fields
| Field | Description |
|---|---|
| `checkpoint_used` | Path to student model checkpoint |
| `generation_number` | Integer generation ID (2) |
| `parent_student` | `"generation_1"` |
| `sampling_strategy` | All generation hyperparameters |
| `seed` | Random seed |
| `total_prompts` | Number of prefix inputs |
| `successful_generations` | Successfully generated records |
| `failed_generations` | Failed batches |
| `average_output_char_length` | Avg generated text length |

---

## 4. Colab Workflow

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount("/content/drive")

# 2. Set path to project
import sys
sys.path.insert(0, "/content/drive/MyDrive/collapse-resistant-training")

# 3. Run generation
from recursive_generation.run_recursive_generation import run_recursive_generation
run_recursive_generation()

# 4. If Colab disconnects — simply re-run the same command.
#    The ResumeManager will automatically skip completed prompts.
```

### Colab Configuration Tips
- Set `batch_size = 16` for T4 GPU with `max_new_tokens = 128`.
- Set `use_amp = True` for float16 half-precision inference.
- The module checkpoints every `checkpoint_every = 500` generations by default.
- All outputs stream directly to JSONL — no RAM accumulation of full dataset.

---

## 5. Checkpoint & Resume

Every `checkpoint_every` (default: 500) successful generations, the `ResumeManager` saves a `resume_state.json` containing the completed prompt indices. On Colab restart:

1. The engine loads `resume_state.json`.
2. It identifies all already-completed indices.
3. It skips those prompts entirely and resumes from the next pending prefix.
4. The JSONL output file is opened in **append mode** so prior records are preserved.

---

## 6. Limitations

- Requires a trained Generation-1 student checkpoint in `checkpoints/student_model/best/`.
- Generation quality depends on student model training convergence.
- Very high risk SCRS scores may indicate the student's outputs will diverge significantly from anchor distributions.
