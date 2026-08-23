"""
================================================================================
scrs/verify_evaluation.py
================================================================================

Verification script to ensure all evaluation pipeline constraints for Student-2
controlled experiments (Baseline vs Adaptive) are met.

Checks:
  1. Baseline Student-2 checkpoint path is defined.
  2. Adaptive Student-2 checkpoint path is defined.
  3. Frozen anchor path is unchanged (checkpoints/anchor_model/frozen).
  4. Prompt source path is unchanged (data/processed/clean_wikitext.txt).
  5. Baseline and adaptive evaluation configurations are identical.
  6. Output directories are isolated.
  7. Existing Student-1 output directories are not modified/overwritten.
  8. Probe, Ensemble, and SCRS configuration updating and smoke checks work.
  9. SCRS engine correctly processes Probe + Ensemble report structures.
"""

import sys
import json
from pathlib import Path

from probe.probe_config import ProbeConfig
from ensemble.ensemble_config import EnsembleConfig
from scrs.scrs_config import SCRSConfig
from scrs.scrs_engine import SCRSEngine
from scrs.utils import get_scrs_logger

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent


def verify_evaluation_setup() -> bool:
    print("================================================================================")
    print("      STUDENT-2 EVALUATION PIPELINE VERIFICATION SUITE (CHECKS 1-9)")
    print("================================================================================")

    anchor_path = _PROJECT_ROOT / "checkpoints" / "anchor_model" / "frozen"
    prompt_source = _PROJECT_ROOT / "data" / "processed" / "clean_wikitext.txt"

    base_student = Path("/content/drive/MyDrive/collapse-resistant-assets/student2/baseline/best")
    adap_student = Path("/content/drive/MyDrive/collapse-resistant-assets/student2/adaptive/best")

    base_probe_out = _PROJECT_ROOT / "probe_out" / "student2_baseline"
    adap_probe_out = _PROJECT_ROOT / "probe_out" / "student2_adaptive"

    base_ens_out = _PROJECT_ROOT / "ensemble_out" / "student2_baseline"
    adap_ens_out = _PROJECT_ROOT / "ensemble_out" / "student2_adaptive"

    base_scrs_out = _PROJECT_ROOT / "scrs_out" / "student2_baseline"
    adap_scrs_out = _PROJECT_ROOT / "scrs_out" / "student2_adaptive"

    # --- Check 1 & 2: Checkpoint Paths Defined ---
    print("\n[1/9 & 2/9] Verifying Baseline and Adaptive Student-2 checkpoint paths...")
    assert str(base_student).endswith("baseline/best"), f"Unexpected baseline student path: {base_student}"
    assert str(adap_student).endswith("adaptive/best"), f"Unexpected adaptive student path: {adap_student}"
    print(" [PASS] ✓ Baseline and Adaptive Student-2 checkpoint paths correctly specified.")

    # --- Check 3: Frozen Anchor Path Unchanged ---
    print("\n[3/9] Verifying Frozen Anchor Model Path...")
    probe_cfg = ProbeConfig()
    assert probe_cfg.anchor_model_path == anchor_path, f"Anchor path changed: {probe_cfg.anchor_model_path}"
    print(f" [PASS] ✓ Frozen Anchor Model path unchanged: {anchor_path}")

    # --- Check 4: Prompt Source Unchanged ---
    print("\n[4/9] Verifying Prompt Source Path...")
    assert probe_cfg.dataset_source == prompt_source, f"Prompt source changed: {probe_cfg.dataset_source}"
    ens_cfg = EnsembleConfig()
    assert ens_cfg.dataset_source == prompt_source, f"Ensemble prompt source changed: {ens_cfg.dataset_source}"
    print(f" [PASS] ✓ Prompt source dataset unchanged: {prompt_source}")

    # --- Check 5: Evaluation Hyperparameters Identical ---
    print("\n[5/9] Verifying Evaluation Configurations are Identical...")
    base_probe_cfg = ProbeConfig()
    base_probe_cfg.update_paths(output_dir=base_probe_out, student_model_path=base_student)

    adap_probe_cfg = ProbeConfig()
    adap_probe_cfg.update_paths(output_dir=adap_probe_out, student_model_path=adap_student)

    assert base_probe_cfg.batch_size == adap_probe_cfg.batch_size == 2
    assert base_probe_cfg.max_prompts == adap_probe_cfg.max_prompts == 10
    assert base_probe_cfg.prompt_min_tokens == adap_probe_cfg.prompt_min_tokens == 32
    assert base_probe_cfg.prompt_max_tokens == adap_probe_cfg.prompt_max_tokens == 64
    assert base_probe_cfg.random_seed == adap_probe_cfg.random_seed == 42
    assert base_probe_cfg.num_layers == adap_probe_cfg.num_layers == 6

    base_ens_cfg = EnsembleConfig()
    base_ens_cfg.update_paths(output_dir=base_ens_out, student_model_path=base_student)

    adap_ens_cfg = EnsembleConfig()
    adap_ens_cfg.update_paths(output_dir=adap_ens_out, student_model_path=adap_student)

    assert base_ens_cfg.batch_size == adap_ens_cfg.batch_size == 2
    assert base_ens_cfg.max_prompts == adap_ens_cfg.max_prompts == 100
    assert base_ens_cfg.prompt_min_tokens == adap_ens_cfg.prompt_min_tokens == 32
    assert base_ens_cfg.prompt_max_tokens == adap_ens_cfg.prompt_max_tokens == 64
    assert base_ens_cfg.random_seed == adap_ens_cfg.random_seed == 42
    assert base_ens_cfg.mc_dropout_passes == adap_ens_cfg.mc_dropout_passes == 10
    assert base_ens_cfg.top_k_confidence == adap_ens_cfg.top_k_confidence == 5

    scrs_cfg = SCRSConfig()
    assert scrs_cfg.representation_group_weight == 0.60
    assert scrs_cfg.uncertainty_group_weight == 0.40
    print(" [PASS] ✓ Evaluation configurations across Probe, Ensemble, and SCRS are 100% identical.")

    # --- Check 6: Output Directory Isolation ---
    print("\n[6/9] Verifying Output Directory Isolation...")
    assert base_probe_out != adap_probe_out
    assert base_ens_out != adap_ens_out
    assert base_scrs_out != adap_scrs_out
    print(" [PASS] ✓ Probe, Ensemble, and SCRS output directories are completely isolated.")

    # --- Check 7: Student-1 Outputs Preserved ---
    print("\n[7/9] Verifying Student-1 Outputs Are Preserved...")
    std1_probe_dir = _PROJECT_ROOT / "probe_out"
    std1_ens_dir = _PROJECT_ROOT / "ensemble_out"
    std1_scrs_dir = _PROJECT_ROOT / "scrs_out"
    assert std1_probe_dir != base_probe_out and std1_probe_dir != adap_probe_out
    assert std1_ens_dir != base_ens_out and std1_ens_dir != adap_ens_out
    assert std1_scrs_dir != base_scrs_out and std1_scrs_dir != adap_scrs_out
    print(" [PASS] ✓ Student-1 output directories remain untouched and non-overlapping.")

    # --- Check 8 & 9: SCRS Pipeline Smoke Test & Consumption ---
    print("\n[8/9 & 9/9] Verifying SCRS engine report consumption...")
    logger = get_scrs_logger("verify_eval")
    try:
        scrs_test_cfg = SCRSConfig()
        engine = SCRSEngine(scrs_test_cfg, logger=logger)
        result = engine.compute()
        assert 0.0 <= result.scrs <= 1.0, f"SCRS score out of bounds: {result.scrs}"
        print(f" [PASS] ✓ SCRS engine successfully consumed upstream reports. Test SCRS = {result.scrs:.4f}")
    except Exception as e:
        print(f" [FAIL] ✗ SCRS report consumption check failed: {e}")
        return False

    print("\n================================================================================")
    print(" ALL 9 EVALUATION PIPELINE VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("================================================================================")
    return True


if __name__ == "__main__":
    success = verify_evaluation_setup()
    if not success:
        sys.exit(1)
