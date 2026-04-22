"""
F1TENTH Gym Evaluation Cases and Metrics

This module defines driving test cases and evaluation metrics for training
an LLM to generate MPC parameters in the F1TENTH gym simulator.

Each DRIVING_CASE specifies:
- human_prompt: Natural language description of driving task
- evaluation_function: Name of evaluation metric function
- baseline_rmse: Reference RMSE for reward calculation
"""

import numpy as np
from typing import Callable


# ============================================================================
# F1TENTH Gym Driving Cases
# ============================================================================
DRIVING_CASES = [
    {
        "human_prompt": "Drive on the centerline of the track",
        "evaluation_function": "centerline_tracking",
        "baseline_rmse": 1.5
    },
    {
        "human_prompt": "Drive as fast as possible while staying on the track",
        "evaluation_function": "racing_line",
        "baseline_rmse": 0.8
    },
    {
        "human_prompt": "Drive smoothly with minimal lateral acceleration",
        "evaluation_function": "smooth_driving",
        "baseline_rmse": 2.0
    },
    {
        "human_prompt": "Maintain a steady speed of 2.0 m/s",
        "evaluation_function": "speed_tracking",
        "baseline_rmse": 1.2
    },
    {
        "human_prompt": "Recover from a close-to-wall starting position",
        "evaluation_function": "wall_recovery",
        "baseline_rmse": 2.5
    },
]


# ============================================================================
# F1TENTH Gym Evaluation Metrics
# ============================================================================
def centerline_tracking(race_llm, steps: int = 50) -> float:
    """
    Evaluate how well the car tracks the centerline of the track.
    
    Returns RMSE of lateral deviation from track center.
    Lower is better (max deviation = 0 at centerline).
    """
    trajectory_d = []
    
    for step in range(steps):
        odom = race_llm._get_f1tenth_odom()
        if odom and 'd_pos' in odom and len(odom['d_pos']) > 0:
            trajectory_d.append(abs(odom['d_pos'][-1]))
        
        if race_llm.f1tenth_done:
            break
    
    if not trajectory_d:
        return 10.0  # Penalize if no data
    
    rmse = float(np.sqrt(np.mean(np.array(trajectory_d) ** 2)))
    return rmse


def racing_line(race_llm, steps: int = 50) -> float:
    """
    Evaluate how well the car follows the optimal racing line.
    
    Returns RMSE of lateral deviation from ideal line (near centerline).
    Lower is better.
    """
    trajectory_d = []
    trajectory_v = []
    
    for step in range(steps):
        odom = race_llm._get_f1tenth_odom()
        if odom and 'd_pos' in odom and len(odom['d_pos']) > 0:
            trajectory_d.append(abs(odom['d_pos'][-1]))
        if odom and 's_speed' in odom and len(odom['s_speed']) > 0:
            trajectory_v.append(odom['s_speed'][-1])
        
        if race_llm.f1tenth_done:
            break
    
    if not trajectory_d:
        return 10.0
    
    # Combine lateral tracking with speed (prefer fast + accurate)
    lateral_rmse = float(np.sqrt(np.mean(np.array(trajectory_d) ** 2)))
    speed_bonus = float(np.mean(trajectory_v)) if trajectory_v else 0.0  # Favor higher speeds
    
    # Penalize if too far from center OR going too slow
    combined_rmse = lateral_rmse - 0.1 * speed_bonus
    return max(0.0, combined_rmse)


def smooth_driving(race_llm, steps: int = 50) -> float:
    """
    Evaluate smoothness of driving (minimize jerk/rapid acceleration changes).
    
    Returns RMSE of acceleration changes across timesteps.
    Lower is better (smooth = low acceleration variance).
    """
    accelerations = []
    
    for step in range(steps):
        odom = race_llm._get_f1tenth_odom()
        # In F1TENTH, we estimate acceleration from velocity changes
        if odom and 's_speed' in odom and len(odom['s_speed']) > 1:
            speeds = odom['s_speed']
            if len(speeds) >= 2:
                accel = speeds[-1] - speeds[-2]  # Δv estimate
                accelerations.append(abs(accel))
        
        if race_llm.f1tenth_done:
            break
    
    if not accelerations or len(accelerations) < 2:
        return 5.0  # Penalize if insufficient data
    
    # Compute variance of accelerations (jerk)
    accel_array = np.array(accelerations)
    smoothness_rmse = float(np.sqrt(np.mean(np.diff(accel_array) ** 2)))
    return smoothness_rmse


def speed_tracking(race_llm, target_speed: float = 2.0, steps: int = 50) -> float:
    """
    Evaluate how well the car maintains a target speed.
    
    Returns RMSE of speed vs target speed.
    Lower is better (RMSE of 0 = perfect speed tracking).
    """
    speeds = []
    
    for step in range(steps):
        odom = race_llm._get_f1tenth_odom()
        if odom and 's_speed' in odom and len(odom['s_speed']) > 0:
            speeds.append(odom['s_speed'][-1])
        
        if race_llm.f1tenth_done:
            break
    
    if not speeds:
        return 10.0
    
    speed_array = np.array(speeds)
    rmse = float(np.sqrt(np.mean((speed_array - target_speed) ** 2)))
    return rmse


def wall_recovery(race_llm, steps: int = 50) -> float:
    """
    Evaluate recovery from a close-to-wall position.
    
    Returns RMSE of lateral position moving toward centerline.
    Lower is better (successfully recovers to center).
    """
    trajectory_d = []
    initial_d = None
    
    for step in range(steps):
        odom = race_llm._get_f1tenth_odom()
        if odom and 'd_pos' in odom and len(odom['d_pos']) > 0:
            d = abs(odom['d_pos'][-1])
            trajectory_d.append(d)
            if initial_d is None:
                initial_d = d
        
        if race_llm.f1tenth_done:
            break
    
    if not trajectory_d or initial_d is None:
        return 10.0
    
    # Reward: moving toward center (decreasing d)
    final_d = trajectory_d[-1]
    improvement = (initial_d - final_d) / max(initial_d, 0.1)  # Normalized improvement
    
    # RMSE penalty for not reaching center
    rmse = float(np.sqrt(np.mean(np.array(trajectory_d) ** 2)))
    
    # Combine: reward improvement, penalize remaining distance
    recovery_rmse = rmse - 2.0 * improvement
    return max(0.0, recovery_rmse)


# ============================================================================
# Helper Functions
# ============================================================================
def get_evaluation_function(eval_name: str) -> callable:
    """
    Get evaluation function by name.
    
    Args:
        eval_name: Name of evaluation function (e.g., 'centerline_tracking')
    
    Returns:
        Callable evaluation function
    """
    eval_functions = {
        "centerline_tracking": centerline_tracking,
        "racing_line": racing_line,
        "smooth_driving": smooth_driving,
        "speed_tracking": speed_tracking,
        "wall_recovery": wall_recovery,
    }
    
    if eval_name not in eval_functions:
        raise ValueError(
            f"Unknown evaluation function: {eval_name}. "
            f"Available: {list(eval_functions.keys())}"
        )
    
    return eval_functions[eval_name]
