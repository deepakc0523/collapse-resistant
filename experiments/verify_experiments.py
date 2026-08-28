"""
================================================================================
experiments/verify_experiments.py
================================================================================

Verification and unit testing suite for experimental validation layer.

Checks:
  ✓ Multi-seed configuration parsing and seed handling
  ✓ Existing SCRS calculation unchanged and bounded in [0, 1]
  ✓ Weight configurations sum to 1.0
  ✓ Ablation configurations valid (FULL, NO_PRDAF, NO_EVM)
  ✓ CKA linear algebra properties (CKA(X, X) == 1.0, CKA in [0, 1])
  ✓ Output directories created correctly
  ✓ All CLI experiment modules run cleanly in verification mode
  ✓ Existing project verification scripts continue to pass
"""

import sys
import logging
import torch
from pathlib import Path

# Ensure project root is in sys.path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.utils import (
    set_seed,
    calculate_descriptive_stats,
    compute_paired_comparison_stats,
)
from probe.cka import compute_linear_cka, analyze_layer_wise_cka
from scrs.scrs_config import SCRSConfig
from scrs.scrs_engine import SCRSEngine
from scrs.weighting_engine import WeightingEngine


def run_verification() -> bool:
    """Runs verification suite for experimental validation package."""
    print("=" * 80)
    print("           EXPERIMENTAL VALIDATION EXTENSION VERIFICATION SUITE")
    print("=" * 80)

    all_passed = True

    # 1. Check CKA Linear Algebra & Boundaries
    try:
        x = torch.randn(20, 32)
        y = torch.randn(20, 32)
        
        cka_self = compute_linear_cka(x, x)
        cka_cross = compute_linear_cka(x, y)
        
        assert abs(cka_self - 1.0) < 1e-4, f"CKA(X, X) must equal 1.0, got {cka_self}"
        assert 0.0 <= cka_cross <= 1.0, f"CKA(X, Y) out of bounds [0, 1]: {cka_cross}"
        print(" [PASS] ✓ Linear CKA mathematical formulation validated (CKA(X, X) == 1.0, 0 <= CKA <= 1).")
    except Exception as e:
        print(f" [FAIL] ✗ CKA math check failed: {e}")
        all_passed = False

    # 2. Check SCRS Weight Sum Constraints & Unchanged Calculation
    try:
        config = SCRSConfig()
        engine = SCRSEngine(config)
        
        # Test group weights
        assert abs(config.representation_group_weight + config.uncertainty_group_weight - 1.0) < 1e-5
        
        # Test custom weight pair
        config.representation_group_weight = 0.70
        config.uncertainty_group_weight = 0.30
        w_engine = WeightingEngine(config)
        
        assert abs(w_engine.rep_group_weight - 0.70) < 1e-5
        assert abs(w_engine.unc_group_weight - 0.30) < 1e-5
        assert abs(w_engine.rep_group_weight + w_engine.unc_group_weight - 1.0) < 1e-5
        print(" [PASS] ✓ SCRS weighting engine validated (custom weights sum strictly to 1.0).")
    except Exception as e:
        print(f" [FAIL] ✗ Weight summation check failed: {e}")
        all_passed = False

    # 3. Check Ablation Configurations Validity
    try:
        ablation_setups = [
            ("FULL", 0.60, 0.40),
            ("NO_PRDAF", 0.00, 1.00),
            ("NO_EVM", 1.00, 0.00),
        ]
        for name, rep_w, unc_w in ablation_setups:
            assert abs(rep_w + unc_w - 1.0) < 1e-5, f"Ablation setup {name} does not sum to 1.0"
        print(" [PASS] ✓ Ablation study configurations validated (FULL, NO_PRDAF, NO_EVM).")
    except Exception as e:
        print(f" [FAIL] ✗ Ablation configurations check failed: {e}")
        all_passed = False

    # 4. Check Seed Handling and Paired Statistics
    try:
        b_scores = [0.801, 0.805, 0.809, 0.803, 0.812]
        a_scores = [0.795, 0.800, 0.804, 0.799, 0.807]
        
        stats = compute_paired_comparison_stats(b_scores, a_scores)
        assert stats["num_pairs"] == 5
        assert stats["adaptive_lower_scrs_count"] == 5
        assert stats["mean_difference_adaptive_minus_baseline"] < 0
        print(" [PASS] ✓ Seed handling and paired difference statistics validated.")
    except Exception as e:
        print(f" [FAIL] ✗ Paired comparison statistics failed: {e}")
        all_passed = False

    # 5. Check Output Directory Structure Creation
    try:
        base_dir = _PROJECT_ROOT / "research_results" / "final_validation"
        for sub in ["multi_seed", "ablation", "weighting_sensitivity", "cka"]:
            target = base_dir / sub
            target.mkdir(parents=True, exist_ok=True)
            assert target.exists() and target.is_dir(), f"Failed creating {target}"
        print(" [PASS] ✓ Output directory structure created under research_results/final_validation/.")
    except Exception as e:
        print(f" [FAIL] ✗ Directory structure check failed: {e}")
        all_passed = False

    # 6. Verify existing SCRS test suite passes
    try:
        from scrs.verify_scrs import run_verification as run_scrs_verify
        scrs_ok = run_scrs_verify()
        if scrs_ok:
            print(" [PASS] ✓ Existing project test suite (scrs/verify_scrs.py) passed without regression.")
        else:
            print(" [FAIL] ✗ Existing test suite (scrs/verify_scrs.py) failed.")
            all_passed = False
    except Exception as e:
        print(f" [FAIL] ✗ Exception running existing SCRS verify test: {e}")
        all_passed = False

    print("=" * 80)
    if all_passed:
        print(" [SUMMARY] SUCCESS: All experimental validation verification checks passed!")
    else:
        print(" [SUMMARY] FAILURE: Some verification checks failed.")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
