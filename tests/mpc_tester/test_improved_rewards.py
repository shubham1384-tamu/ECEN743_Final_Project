#!/usr/bin/env python3
"""
Test Script: Compare Original vs Improved Reward Functions

This script tests both reward implementations on synthetic LLM outputs
without requiring a full training run.

Usage:
    python test_improved_rewards.py [--compare] [--detailed]
"""

import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import argparse
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────
# Test Data: Synthetic LLM Outputs
# ─────────────────────────────────────────────────────────────────────────

SYNTHETIC_OUTPUTS = [
    # Test 1: Perfect output (XML + valid dict)
    {
        "name": "Perfect Output",
        "output": """<reasoning>
The car should drive forward, so we increase v_max for speed.
</reasoning>
<answer>
new_mpc_params = {
    'qv': 15.0,
    'qn': 22.0,
    'qalpha': 8.0,
    'qac': 0.05,
    'qddelta': 0.15,
    'alat_max': 10.5,
    'a_min': -9.0,
    'a_max': 11.0,
    'v_min': 1.5,
    'v_max': 13.0,
    'track_safety_margin': 0.4
}
</answer>""",
        "expected_format": 1.0,
        "expected_extraction": 1.0,
        "expected_params_valid": True,
    },
    
    # Test 2: Missing close tag
    {
        "name": "Missing Close Tag",
        "output": """<reasoning>
Tuning for reverse driving
<answer>
new_mpc_params = {
    'qv': 20.0,
    'qn': 25.0,
    'v_max': -1.0,
    'v_min': -5.0
}
</answer>""",
        "expected_format": 0.7,  # Graduated: one tag missing
        "expected_extraction": 0.7,  # Partial extraction
        "expected_params_valid": False,
    },
    
    # Test 3: Out-of-range parameters
    {
        "name": "Out-of-Range Parameters",
        "output": """<reasoning>
Ultra-aggressive tuning
</reasoning>
<answer>
new_mpc_params = {
    'qv': 500.0,
    'qn': 100.0,
    'v_max': 50.0,
    'v_min': -20.0
}
</answer>""",
        "expected_format": 1.0,
        "expected_extraction": 0.8,  # Partial extraction (4/11 params)
        "expected_params_valid": False,
        "expected_range_score": 0.0,  # All out of range
    },
    
    # Test 4: No XML tags (fallback format)
    {
        "name": "No XML Tags",
        "output": """new_mpc_params = {
    'qv': 12.0,
    'qn': 18.0,
    'qalpha': 7.5,
    'v_max': 10.0
}""",
        "expected_format": 0.4,  # Graduated: no tags but dict present
        "expected_extraction": 0.6,  # Partial (4/11)
        "expected_params_valid": False,
    },
    
    # Test 5: Empty or malformed
    {
        "name": "Malformed Dictionary",
        "output": """<reasoning>Invalid attempt</reasoning>
<answer>
new_mpc_params = {
    'qv': twenty,
    'qn': invalid
}
</answer>""",
        "expected_format": 1.0,
        "expected_extraction": 0.0,  # Can't parse
        "expected_params_valid": False,
    },
]


# ─────────────────────────────────────────────────────────────────────────
# Mock RaceLLMMPC for Testing
# ─────────────────────────────────────────────────────────────────────────

class MockRaceLLMMPC:
    """Mock implementation that doesn't require F1TENTH gym."""
    
    DEFAULT_MPC_PARAMS = {
        "qv": 10.0, "qn": 20.0, "qalpha": 7.0, "qac": 0.01,
        "qddelta": 0.1, "alat_max": 10.0, "a_min": -10.0,
        "a_max": 10.0, "v_min": 1.0, "v_max": 12.0,
        "track_safety_margin": 0.3
    }
    
    def __init__(self):
        self.f1tenth_done = False
    
    def _sanitize_tune_output(self, text: str) -> Tuple[Dict, str]:
        """Parse MPC parameters from text (same as real implementation)."""
        import re
        import ast
        
        command_dict = None
        explanation_text = text
        
        try:
            # Extract answer tag content if present (XML format)
            answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
            if answer_match:
                content = answer_match.group(1)
            else:
                content = text
            
            # Find "new_mpc_params = {" and extract the matching dictionary
            match = re.search(r'new_mpc_params\s*=\s*({.*})', content, re.DOTALL)
            if match:
                dict_str = match.group(1)
                try:
                    dict_str_clean = re.sub(r'\s+', ' ', dict_str)
                    command_dict = ast.literal_eval(dict_str_clean)
                    
                    if isinstance(command_dict, dict) and len(command_dict) > 0:
                        return command_dict, explanation_text
                except (ValueError, SyntaxError):
                    pass
        except Exception:
            pass
        
        return command_dict, explanation_text


# ─────────────────────────────────────────────────────────────────────────
# Test Runner
# ─────────────────────────────────────────────────────────────────────────

def test_format_reward():
    """Test graduated format reward scoring."""
    from train.improved_reward_functions import ImprovedRewardFunctions
    
    mock_llm = MockRaceLLMMPC()
    rewards_obj = ImprovedRewardFunctions(mock_llm, None)
    
    print("\n" + "=" * 70)
    print("TEST 1: FORMAT REWARD (Graduated Scoring)")
    print("=" * 70)
    
    for test_case in SYNTHETIC_OUTPUTS:
        score = rewards_obj._check_tags_graduated(test_case["output"], "reasoning")
        score += rewards_obj._check_tags_graduated(test_case["output"], "answer")
        score /= 2.0
        
        expected = test_case.get("expected_format", 0.5)
        status = "✓" if abs(score - expected) < 0.1 else "⚠"
        
        print(f"{status} {test_case['name']:30} → {score:.2f} (expected ~{expected:.2f})")


def test_extraction_reward():
    """Test graduated extraction reward."""
    from train.improved_reward_functions import ImprovedRewardFunctions
    
    mock_llm = MockRaceLLMMPC()
    rewards_obj = ImprovedRewardFunctions(mock_llm, None)
    
    print("\n" + "=" * 70)
    print("TEST 2: EXTRACTION REWARD (Graduated Scoring)")
    print("=" * 70)
    
    for test_case in SYNTHETIC_OUTPUTS:
        extracted, _ = mock_llm._sanitize_tune_output(test_case["output"])
        
        if extracted is None:
            score = 0.0
        elif len(extracted) == 0:
            score = 0.3
        elif len(extracted) < 11:  # 11 total MPC params
            score = 0.5 + 0.5 * (len(extracted) / 11)
        else:
            score = 1.0
        
        expected = test_case.get("expected_extraction", 0.5)
        status = "✓" if abs(score - expected) < 0.15 else "⚠"
        
        param_count = len(extracted) if extracted else 0
        print(f"{status} {test_case['name']:30} → {score:.2f} (params: {param_count}, expected ~{expected:.2f})")


def test_param_range_reward():
    """Test parameter range validation."""
    from train.improved_reward_functions import ImprovedRewardFunctions, MPC_PARAM_RANGES
    
    mock_llm = MockRaceLLMMPC()
    rewards_obj = ImprovedRewardFunctions(mock_llm, None)
    
    print("\n" + "=" * 70)
    print("TEST 3: PARAMETER RANGE REWARD")
    print("=" * 70)
    
    for test_case in SYNTHETIC_OUTPUTS:
        extracted, _ = mock_llm._sanitize_tune_output(test_case["output"])
        
        if not extracted:
            score = 0.0
        else:
            in_range = 0
            for param, value in extracted.items():
                if param in MPC_PARAM_RANGES:
                    min_v, max_v = MPC_PARAM_RANGES[param]
                    if min_v <= value <= max_v:
                        in_range += 1
            score = in_range / len(extracted) if extracted else 0.0
        
        expected = test_case.get("expected_range_score", None)
        if expected is not None:
            status = "✓" if abs(score - expected) < 0.1 else "⚠"
            print(f"{status} {test_case['name']:30} → {score:.2f} (expected {expected:.2f})")
        else:
            print(f"   {test_case['name']:30} → {score:.2f}")


def test_comparison():
    """Compare original vs improved rewards."""
    from train.train_mpc_f1tenth import RewardFunctions
    from train.improved_reward_functions import ImprovedRewardFunctions
    
    mock_llm = MockRaceLLMMPC()
    
    original = RewardFunctions(mock_llm, None)
    improved = ImprovedRewardFunctions(mock_llm, None)
    
    print("\n" + "=" * 70)
    print("TEST 4: ORIGINAL vs IMPROVED - Format Reward Comparison")
    print("=" * 70)
    
    test_outputs = [case["output"] for case in SYNTHETIC_OUTPUTS]
    
    # Test format reward
    orig_format = original.format_reward([""] * len(test_outputs), test_outputs)
    impr_format = improved.format_reward([""] * len(test_outputs), test_outputs)
    
    orig_scores = orig_format[0]
    impr_scores = impr_format[0]
    
    print("\nFormat Reward:")
    print(f"{'Test Case':30} | Original | Improved | Difference")
    print("-" * 65)
    for i, test_case in enumerate(SYNTHETIC_OUTPUTS):
        diff = impr_scores[i] - orig_scores[i]
        status = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
        print(f"{test_case['name']:30} | {orig_scores[i]:8.3f} | {impr_scores[i]:8.3f} | {status} {abs(diff):6.3f}")


def main():
    parser = argparse.ArgumentParser(description="Test Improved Reward Functions")
    parser.add_argument("--compare", action="store_true", help="Run comparison tests")
    parser.add_argument("--detailed", action="store_true", help="Print detailed output")
    args = parser.parse_args()
    
    print("\n" + "🎯 " * 20)
    print("IMPROVED REWARD FUNCTIONS - TEST SUITE")
    print("🎯 " * 20)
    
    # Run tests
    test_format_reward()
    test_extraction_reward()
    test_param_range_reward()
    
    if args.compare:
        test_comparison()
    
    print("\n" + "=" * 70)
    print("✓ All tests completed!")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Review improved_reward_functions.py for implementation details")
    print("2. See INTEGRATION_GUIDE.md for how to use in training")
    print("3. Run training with: python train/train_mpc_f1tenth.py")
    print("\n")


if __name__ == "__main__":
    main()
