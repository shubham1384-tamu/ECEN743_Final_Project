"""
Improved Reward Functions for LLM MPC Training

This module provides enhanced reward functions that address limitations in the original
reward system:
1. Multi-horizon driving evaluation (short-term + long-term stability)
2. Multi-metric driving score (lateral, speed, smoothness, boundary safety)
3. Explicit collision penalty
4. Graduated format/extraction rewards
5. Parameter range validation rewards

Usage:
    # In train_mpc_f1tenth.py, replace RewardFunctions with ImprovedRewardFunctions
    reward_funcs_obj = ImprovedRewardFunctions(
        race_llm=race_llm,
        tokenizer=tokenizer,
        use_wandb=use_wandb,
        w1=1.0,
        w6=1.0,
        # New weights:
        w_lateral=0.40,
        w_speed=0.30,
        w_smoothness=0.20,
        w_boundary=0.10,
    )
"""

import numpy as np
import re
import ast
from typing import Dict, List, Tuple, Optional

# MPC Parameters (same as in train_mpc_f1tenth.py)
MPC_PARAM_NAMES = [
    "qv", "qn", "qalpha", "qac", "qddelta", "alat_max",
    "a_min", "a_max", "v_min", "v_max", "track_safety_margin"
]

# Realistic parameter ranges for bonus reward
MPC_PARAM_RANGES = {
    "qv": (1.0, 100.0),           # Lateral velocity weight
    "qn": (1.0, 50.0),            # Normal deviation weight
    "qalpha": (0.1, 20.0),        # Steering weight
    "qac": (0.001, 1.0),          # Acceleration weight
    "qddelta": (0.01, 10.0),      # Steering rate weight
    "alat_max": (5.0, 15.0),      # Max lateral accel
    "a_min": (-15.0, -1.0),       # Min acceleration
    "a_max": (1.0, 15.0),         # Max acceleration
    "v_min": (0.0, 5.0),          # Min velocity (m/s)
    "v_max": (2.0, 15.0),         # Max velocity (m/s)
    "track_safety_margin": (0.0, 1.0),  # Safety margin
}


class ImprovedRewardFunctions:
    """Enhanced reward functions with multi-horizon and multi-metric evaluation."""
    
    def __init__(self, race_llm, tokenizer, use_wandb=False, w1=1.0, w6=1.0,
                 w_lateral=0.40, w_speed=0.30, w_smoothness=0.20, w_boundary=0.10):
        """
        Initialize improved reward functions.
        
        Args:
            race_llm: RaceLLMMPC instance
            tokenizer: Tokenizer for token counting
            use_wandb: Whether to log to Weights & Biases
            w1: Weight for driving reward
            w6: Weight for format reward
            w_lateral: Weight for lateral tracking in multi-metric (0.40 default)
            w_speed: Weight for speed consistency (0.30 default)
            w_smoothness: Weight for acceleration smoothness (0.20 default)
            w_boundary: Weight for boundary safety (0.10 default)
        """
        self.race_llm = race_llm
        self.tokenizer = tokenizer
        self.use_wandb = use_wandb
        self.w1 = w1
        self.w6 = w6
        
        # Multi-metric weights (must sum to ~1.0)
        self.w_lateral = w_lateral
        self.w_speed = w_speed
        self.w_smoothness = w_smoothness
        self.w_boundary = w_boundary
        
        # Normalize weights
        total = self.w_lateral + self.w_speed + self.w_smoothness + self.w_boundary
        self.w_lateral /= total
        self.w_speed /= total
        self.w_smoothness /= total
        self.w_boundary /= total
        
        self.step = 0
    
    # ─────────────────────────────────────────────────────────────────────
    # FORMAT & EXTRACTION REWARDS (Improved: Graduated instead of Binary)
    # ─────────────────────────────────────────────────────────────────────
    
    def format_reward(self, prompts, completions, **kwargs) -> list:
        """
        Improved format reward with graduated scoring.
        
        Instead of binary (0 or 1), rewards partial XML correctness.
        - Perfect XML: 1.0
        - One tag missing: 0.7
        - Severely malformed: < 0.5
        """
        reasoning_scores = np.zeros(len(prompts))
        answer_scores = np.zeros(len(prompts))
        
        for i, completion in enumerate(completions):
            reasoning_scores[i] = self._check_tags_graduated(completion, "reasoning")
            answer_scores[i] = self._check_tags_graduated(completion, "answer")
        
        reward = (reasoning_scores + answer_scores) / 2.0 * self.w6
        print(f"[Format Reward] Mean: {reward.mean():.3f}, Min: {reward.min():.3f}, Max: {reward.max():.3f}")
        return [reward]
    
    def extraction_reward(self, prompts, completions, **kwargs) -> list:
        """
        Improved extraction reward with graduated scoring.
        
        Rewards partial dict extraction:
        - Fully extracted: 1.0
        - Partially extracted (some params found): 0.5-0.8
        - No extraction: 0.0
        """
        rewards = np.zeros(len(prompts))
        
        for i, completion in enumerate(completions):
            extracted, _ = self.race_llm._sanitize_tune_output(completion)
            
            if extracted is None:
                rewards[i] = 0.0
            elif len(extracted) == 0:
                rewards[i] = 0.3  # Partial: found dict but empty
            elif len(extracted) < len(MPC_PARAM_NAMES):
                # Partial extraction: some params found
                rewards[i] = 0.5 + 0.5 * (len(extracted) / len(MPC_PARAM_NAMES))
            else:
                rewards[i] = 1.0  # All params extracted
        
        print(f"[Extraction Reward] Mean: {rewards.mean():.3f}, Success rate: {(rewards > 0).mean():.1%}")
        return [rewards]
    
    def param_name_reward(self, prompts, completions, **kwargs) -> list:
        """Reward for correctly named MPC parameters (unchanged)."""
        rewards = np.zeros(len(prompts))
        
        for i, completion in enumerate(completions):
            extracted, _ = self.race_llm._sanitize_tune_output(completion)
            if extracted is None:
                rewards[i] = 0.0
                continue
            
            if not extracted:
                rewards[i] = 1.0
                continue
            
            correct = sum(1 for p in extracted.keys() if isinstance(p, str) and p in MPC_PARAM_NAMES)
            rewards[i] = correct / len(extracted)
        
        print(f"[Param Name Reward] Mean: {rewards.mean():.3f}")
        return [rewards]
    
    # ─────────────────────────────────────────────────────────────────────
    # NEW: PARAMETER RANGE REWARD
    # ─────────────────────────────────────────────────────────────────────
    
    def param_range_reward(self, prompts, completions, **kwargs) -> list:
        """
        NEW: Reward parameters within realistic bounds.
        
        Prevents model from suggesting extreme unrealistic values.
        Range: 0.0 (all out-of-range) to 1.0 (all in-range)
        """
        rewards = np.zeros(len(prompts))
        
        for i, completion in enumerate(completions):
            extracted, _ = self.race_llm._sanitize_tune_output(completion)
            if extracted is None or not extracted:
                rewards[i] = 0.0
                continue
            
            in_range_count = 0
            for param_name, param_value in extracted.items():
                if param_name in MPC_PARAM_RANGES:
                    try:
                        param_value = float(param_value)
                        min_val, max_val = MPC_PARAM_RANGES[param_name]
                        if min_val <= param_value <= max_val:
                            in_range_count += 1
                    except (TypeError, ValueError):
                        # Skip None values or values that can't be converted
                        pass
            
            rewards[i] = in_range_count / len(extracted) if extracted else 0.0
        
        print(f"[Param Range Reward] Mean: {rewards.mean():.3f}")
        return [rewards]
    
    # ─────────────────────────────────────────────────────────────────────
    # IMPROVED DRIVING REWARD (Multi-Horizon + Multi-Metric)
    # ─────────────────────────────────────────────────────────────────────
    
    def driving_reward(self, prompts, completions, f1tenth_steps=50, f1tenth_map="vegas",
                       baseline_rmse=2.0, **kwargs) -> list:
        """
        IMPROVED: Multi-horizon and multi-metric driving reward.
        
        Evaluates MPC parameters using:
        1. Short-term horizon (50 steps) for immediate response
        2. Long-term horizon (200 steps) for stability
        3. Multiple metrics: lateral, speed, smoothness, boundary safety
        4. Explicit collision penalty
        
        Returns composite reward balancing all objectives.
        """
        print("=" * 70)
        print("[Driving Reward] IMPROVED: Multi-Horizon + Multi-Metric")
        print("=" * 70)
        
        if isinstance(baseline_rmse, (list, tuple)):
            baseline_rmse = baseline_rmse[0] if baseline_rmse else 2.0
        baseline_rmse = float(baseline_rmse)
        
        rewards = np.zeros(len(prompts))
        
        for i, completion in enumerate(completions):
            # Reset environment
            self.race_llm.reset_f1tenth()
            
            # Extract and validate parameters
            extracted, _ = self.race_llm._sanitize_tune_output(completion)
            if extracted is None:
                print(f"  Sample {i}: Failed to extract parameters → reward = 0")
                rewards[i] = 0.0
                continue
            
            # Sanitize parameter names
            sanitized = self._sanitize_params(extracted)
            full_params = {**self.race_llm.DEFAULT_MPC_PARAMS, **sanitized}
            
            # Validate that all required parameters are non-None
            if not self._validate_params(full_params):
                print(f"  Sample {i}: Invalid parameters (None values) → reward = 0")
                rewards[i] = 0.0
                continue
            
            # ── Short-term horizon (50 steps ≈ 0.5 sec) ──
            trajectory_short = []
            for step in range(f1tenth_steps):
                obs, done = self.race_llm._step_f1tenth(full_params)
                odom = self.race_llm._get_f1tenth_odom()
                trajectory_short.append(odom)  # Store full odom dict
                if done:
                    print(f"  Sample {i}: Crashed at step {step} (short horizon)")
                    rewards[i] = -4.0  # 🔴 COLLISION PENALTY
                    self.race_llm.reset_f1tenth()
                    continue
            
            if rewards[i] < 0:  # Already penalized for crash
                continue
            
            # ── Long-term horizon (200 steps ≈ 2.0 sec) ──
            self.race_llm.reset_f1tenth()
            trajectory_long = []
            for step in range(f1tenth_steps * 4):  # 4x longer horizon
                obs, done = self.race_llm._step_f1tenth(full_params)
                odom = self.race_llm._get_f1tenth_odom()
                trajectory_long.append(odom)
                if done:
                    print(f"  Sample {i}: Crashed at step {step} (long horizon)")
                    rewards[i] = -4.0  # 🔴 COLLISION PENALTY
                    self.race_llm.reset_f1tenth()
                    continue
            
            if rewards[i] < 0:  # Already penalized for crash
                continue
            
            # ── Compute multi-metric reward ──
            short_reward = self._compute_multi_metric_reward(trajectory_short, baseline_rmse)
            long_reward = self._compute_multi_metric_reward(trajectory_long, baseline_rmse * 1.5)
            
            # Weighted combination: 40% short-term, 60% long-term
            # (Long-term stability is more important for racing)
            combined_reward = 0.4 * short_reward + 0.6 * long_reward
            rewards[i] = float(np.clip(combined_reward, -4.0, 4.0))
            
            status = "✓" if rewards[i] > 0 else "✗"
            print(f"  Sample {i}: short={short_reward:.3f}, long={long_reward:.3f}, "
                  f"combined={rewards[i]:.3f} {status}")
        
        print(f"[Driving Reward] Mean: {rewards.mean():.3f}, Min: {rewards.min():.3f}")
        print("=" * 70)
        return [rewards * self.w1]
    
    # ─────────────────────────────────────────────────────────────────────
    # HELPER FUNCTIONS
    # ─────────────────────────────────────────────────────────────────────
    
    def _compute_multi_metric_reward(self, trajectory: List[Dict], baseline_rmse: float) -> float:
        """
        Compute composite reward from multiple metrics.
        
        Metrics:
        1. Lateral tracking (d_pos RMSE)
        2. Speed consistency (velocity variance)
        3. Acceleration smoothness (steering/speed rate of change)
        4. Boundary safety (distance to walls)
        
        Returns: Composite reward in range [-4, 4]
        """
        if not trajectory:
            return 0.0
        
        # Extract metrics from trajectory
        d_pos_list = [abs(odom['d_pos'][-1]) for odom in trajectory if odom]
        s_speed_list = [abs(odom['s_speed'][-1]) for odom in trajectory if odom]
        d_speed_list = [abs(odom['d_speed'][-1]) for odom in trajectory if odom]
        
        # 1. LATERAL TRACKING: Reward lower lateral RMSE
        lateral_rmse = float(np.mean(d_pos_list)) if d_pos_list else 10.0
        lateral_improvement = (baseline_rmse - lateral_rmse) / baseline_rmse
        lateral_score = np.clip(lateral_improvement, -1.0, 1.0)
        
        # 2. SPEED CONSISTENCY: Reward lower speed variance
        # Target: steady speed (low variance). Penalize erratic speed changes.
        speed_variance = np.var(s_speed_list) if len(s_speed_list) > 1 else 0.0
        # Normalize: typical variance ~0.5-2.0, so threshold at 1.0
        speed_consistency = max(0.0, 1.0 - speed_variance / 1.0)
        speed_score = speed_consistency
        
        # 3. ACCELERATION SMOOTHNESS: Reward smooth acceleration
        # Lower d_speed (lateral speed changes) = smoother control
        accel_smoothness = float(np.mean(d_speed_list)) if d_speed_list else 10.0
        # Normalize: typical d_speed ~0.1-0.5, so threshold at 0.3
        accel_score = max(0.0, 1.0 - accel_smoothness / 0.3)
        
        # 4. BOUNDARY SAFETY: Penalize getting too close to walls
        # Extract boundary distances if available from odom
        # For now, we'll use lateral deviation as proxy (already in d_pos)
        # Ideal: d_pos < 0.5m (safe zone), penalize if > 0.8m
        boundary_safety = max(0.0, 1.0 - lateral_rmse / 0.8)
        
        # Composite score with weights
        composite = (
            self.w_lateral * lateral_score +
            self.w_speed * speed_score +
            self.w_smoothness * accel_score +
            self.w_boundary * boundary_safety
        )
        
        # Scale to [-4, 4] range for consistency with clipping
        composite_reward = composite * 4.0 - 2.0  # [-4, 4] range
        
        return float(composite_reward)
    
    @staticmethod
    def _check_tags_graduated(text: str, tag: str) -> float:
        """
        Check XML tag formatting with graduated scoring.
        
        Returns:
        - 1.0: Both tags present and balanced
        - 0.7: One tag missing
        - 0.4: Both tags missing but dict structure present
        - 0.0: No tags and no dict
        """
        open_count = text.count(f"<{tag}>")
        close_count = text.count(f"</{tag}>")
        
        if open_count == close_count == 1:
            return 1.0  # Perfect
        elif (open_count == 1 and close_count == 0) or (open_count == 0 and close_count == 1):
            return 0.7  # One tag missing
        elif open_count == 0 and close_count == 0:
            # Check if dict structure is present (partial credit)
            if "new_mpc_params" in text and "{" in text:
                return 0.4
            return 0.0
        else:
            # Multiple/mismatched tags
            mismatch = abs(open_count - close_count)
            return max(0.0, 1.0 - 0.2 * mismatch)
    
    @staticmethod
    def _sanitize_params(extracted: dict) -> dict:
        """Sanitize parameter names and values. Convert to float."""
        sanitized = {}
        for param, value in extracted.items():
            if param in MPC_PARAM_NAMES:
                try:
                    # Ensure value is float (LLM may return strings)
                    sanitized[param] = float(value)
                except (TypeError, ValueError):
                    # Skip invalid values
                    pass
        return sanitized
    
    @staticmethod
    def _validate_params(params: dict) -> bool:
        """Validate that all parameters are non-None and numeric."""
        for key, value in params.items():
            if value is None:
                return False
            try:
                float(value)
            except (TypeError, ValueError):
                return False
        return True


# ─────────────────────────────────────────────────────────────────────────
# UTILITY: Legacy wrapper for backwards compatibility
# ─────────────────────────────────────────────────────────────────────────

def create_improved_rewards(race_llm, tokenizer, config: dict = None, use_wandb=False) -> ImprovedRewardFunctions:
    """
    Factory function to create improved reward functions from config.
    
    Args:
        race_llm: RaceLLMMPC instance
        tokenizer: Tokenizer
        config: Optional config dict with keys:
                - w1, w6 (reward weights)
                - w_lateral, w_speed, w_smoothness, w_boundary (metric weights)
        use_wandb: Whether to use Weights & Biases
    
    Returns:
        ImprovedRewardFunctions instance
    """
    config = config or {}
    
    return ImprovedRewardFunctions(
        race_llm=race_llm,
        tokenizer=tokenizer,
        use_wandb=use_wandb,
        w1=config.get("w1", 1.0),
        w6=config.get("w6", 1.0),
        w_lateral=config.get("w_lateral", 0.40),
        w_speed=config.get("w_speed", 0.30),
        w_smoothness=config.get("w_smoothness", 0.20),
        w_boundary=config.get("w_boundary", 0.10),
    )
