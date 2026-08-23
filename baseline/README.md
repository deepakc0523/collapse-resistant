# Student-2 Control Baseline Dataset Builder (`baseline/`)

The **Student-2 Control Baseline Dataset Builder** constructs the **control condition** training dataset for Student-2 in the **Collapse-Resistant Recursive Language Model Training Framework**.

---

## 1. Experimental Purpose

To evaluate the effectiveness of the proposed SCRS $\to$ ATE $\to$ Curriculum Controller framework, we compare two experimental conditions for Generation-2 (Student-2):

| Condition | Experiment ID | Dataset Composition | Curriculum / Policy | Output Location |
|---|---|---|---|---|
| **CONTROL** | `uncontrolled_recursive_baseline` | **100% Generation-2 Synthetic** (0% Human Anchor) | None (Uncontrolled baseline) | `baseline_out/generation_2/` |
| **PROPOSED** | `collapse_resistant_adaptive` | **75% Human Anchor / 25% Synthetic** | 3-Stage Progressive Schedule | `curriculum_out/generation_2/` |

---

## 2. Source Data Specification

- **Source File**: `data/synthetic/generation_2/generation_2_synthetic.jsonl`
- **Source Model**: Student-1 (`parent_student = "generation_1"`)
- **Generation ID**: Generation-2 (`generation = 2`)
- **Total Source Records**: 1,000 deterministically generated records

---

## 3. Directory Structure

```
baseline/
├── __init__.py
├── baseline_config.py
├── utils.py
├── baseline_loader.py
├── baseline_validator.py
├── baseline_exporter.py
├── metadata_generator.py
├── baseline_report.py
├── run_baseline.py
├── verify_baseline.py
└── README.md
```

---

## 4. Output Specification (`baseline_out/generation_2/`)

- `train.jsonl`: 900 samples (90% training split)
- `validation.jsonl`: 100 samples (10% validation split)
- `metadata.json`: Experiment metadata payload
- `baseline_summary.txt`: Human-readable summary report

### `metadata.json` Payload Schema

```json
{
    "experiment": "uncontrolled_recursive_baseline",
    "generation_id": "generation_2",
    "source_generation": 2,
    "parent_student": "generation_1",
    "synthetic_source": "data/synthetic/generation_2/generation_2_synthetic.jsonl",
    "synthetic_source_record_count": 1000,
    "synthetic_ratio": 1.0,
    "anchor_ratio": 0.0,
    "random_seed": 42,
    "dataset_sizes": {
        "total_samples": 1000,
        "train_samples": 900,
        "val_samples": 100,
        "synthetic_count": 1000,
        "anchor_count": 0
    }
}
```

---

## 5. Execution & Verification

To run the verification suite (Checks 1–10):
```bash
python -m baseline.verify_baseline
```

To run the baseline dataset builder:
```bash
python -m baseline.run_baseline
```
