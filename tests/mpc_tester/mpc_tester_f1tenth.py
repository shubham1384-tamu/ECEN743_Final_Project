"""
Simulation-based MPC Tester for F1TENTH Gym

This module provides evaluation of LLM-tuned MPC parameters using the F1TENTH
gym simulator (no ROS required). It tests parameter tuning effectiveness across
multiple driving scenarios: centerline tracking, reference velocity, reverse, 
and smooth driving.

Usage:
    python mpc_tester_f1tenth.py --model-dir <path_to_model> --model custom
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Add parent directories to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llm_mpc_render import RaceLLMMPC
from inference.inf_pipeline import CHAT_TEMPLATE_OPTIONS
from train.utils.mpc.eval_cases import EVAL_DRIVING_CASES


TEST_OPTIONS = ['center', 'refvel', 'reverse', 'smooth']


class SimTester:
    """
    Simulation-based tester for evaluating LLM-tuned MPC parameters.
    Uses F1TENTH gym environment (no ROS required).
    """
    
    def __init__(self, model_type, model_dir, quant, run_name, host_ip='127.0.0.1', render=False, debug=False):
        """
        Initialize the simulation tester.
        
        Args:
            model_type: 'custom', 'gpt-4o', or 'training'
            model_dir: Path to model directory
            quant: Whether to use quantization
            run_name: Name for this test run
            host_ip: Ignored for simulation (kept for API compatibility)
            render: Enable visualization (requires display; disabled by default for headless systems)
            debug: Enable debug logging for troubleshooting
        """
        self.render_enabled = render
        self.debug_enabled = debug
        
        self.racechat = RaceLLMMPC(
            openai_token=os.getenv("OPENAI_API_TOKEN", ""),
            model=model_type,
            model_dir=model_dir,
            quant=quant,
            no_ROS=True,  # Use F1TENTH gym, not ROS
            use_f1tenth=True,
            f1tenth_map='vegas',
            render_sim=render,  # Enable/disable visualization based on flag
            render_mode='human',
            sim_rollout_steps=100  # Steps to run simulation per LLM update
        )
        
        self.model_name = model_type
        self.run_name = run_name
        self.timeout = 60  # seconds
        
        # Create eval directory
        self.eval_dir = os.path.join('tests/mpc_tester/eval', run_name)
        os.makedirs(self.eval_dir, exist_ok=True)
        
        # Default MPC parameters
        self.default_mpc_params = {
            "qv": 10.0,
            "qn": 20.0,
            "qalpha": 7.0,
            "qac": 0.01,
            "qddelta": 0.1,
            "alat_max": 10.0,
            "a_min": -10.0,
            "a_max": 10.0,
            "v_min": 1.0,
            "v_max": 12.0,
            "track_safety_margin": 0.3
        }
        
        # Target speeds for tests
        self.rev_target_speed = -1.0  # m/s for reverse
        self.target_speed = 1.25  # m/s for forward driving
    
    def reset_mpc_params(self):
        """Reset MPC parameters to defaults."""
        self.racechat.current_params = self.default_mpc_params.copy()
    
    def run_sim_episode(self, prompt, timeout=60):
        """
        Run a single simulation episode with LLM-selected parameters.
        
        Args:
            prompt: Natural language driving command
            timeout: Maximum simulation time (seconds)
            
        Returns:
            Dictionary with episode data: {'command': params, 'explanation': str, 'trajectory': dict}
        """
        print(f"Running episode: {prompt}")
        
        # Get LLM command
        start_time = time.time()
        try:
            command, explanation, mem_sources, _, _ = self.racechat.race_mpc_interact(
                scenario=prompt,
                memory_nb=5
            )
            llm_inf_latency = time.time() - start_time
        except Exception as e:
            print(f"LLM inference error: {e}")
            return {
                'command': None,
                'explanation': str(e),
                'llm_inf_latency': time.time() - start_time,
                'trajectory': {},
                'mem_sources': []
            }
        
        # Apply parameters and collect trajectory
        trajectory = self.collect_trajectory(mpc_params=command, timeout=timeout)
        
        return {
            'command': command,
            'explanation': explanation,
            'llm_inf_latency': llm_inf_latency,
            'trajectory': trajectory,
            'mem_sources': mem_sources
        }
    
    def collect_trajectory(self, mpc_params=None, timeout=60, debug=False):
        """
        Collect trajectory data from simulation.
        
        Args:
            mpc_params: MPC parameters dict to apply during trajectory collection
            timeout: Maximum simulation time
            debug: Print debug info about observation format
            
        Returns:
            Dictionary with trajectory data
        """
        trajectory = {
            's_pos': [],
            'd_pos': [],
            's_speed': [],
            'd_speed': [],
            'ax': [],
            'ay': [],
            'time': []
        }
        
        if mpc_params is None:
            mpc_params = self.default_mpc_params.copy()
        
        start_time = time.time()
        obs_format_logged = False
        
        # Render only if enabled (for visualization on systems with display)
        if self.render_enabled:
            try:
                self.racechat.env.render()
            except Exception as e:
                print(f"[WARN] Rendering failed (headless system?): {e}")
        
        # Run simulation
        while time.time() - start_time < timeout:
            # Get current observation
            if self.racechat.obs is not None:
                try:
                    # Log observation format once for debugging
                    if not obs_format_logged and debug:
                        print(f"[DEBUG] Observation type: {type(self.racechat.obs)}")
                        if isinstance(self.racechat.obs, dict):
                            print(f"[DEBUG] Observation keys: {self.racechat.obs.keys()}")
                        else:
                            print(f"[DEBUG] Observation shape: {self.racechat.obs.shape if hasattr(self.racechat.obs, 'shape') else 'N/A'}")
                        obs_format_logged = True
                    
                    # Extract data from observation
                    if isinstance(self.racechat.obs, dict):
                        # F1TENTH gym returns dict with individual fields:
                        # 'poses_x', 'poses_y', 'poses_theta', 'linear_vels_x', 'linear_vels_y'
                        try:
                            x = float(self.racechat.obs['poses_x'][0])
                            y = float(self.racechat.obs['poses_y'][0])
                            theta = float(self.racechat.obs['poses_theta'][0])
                            vx = float(self.racechat.obs['linear_vels_x'][0])
                            vy = float(self.racechat.obs['linear_vels_y'][0])
                            ax, ay = 0.0, 0.0
                        except KeyError as ke:
                            if debug:
                                print(f"[DEBUG] KeyError parsing observation: {ke}")
                                print(f"[DEBUG] Available keys: {list(self.racechat.obs.keys())}")
                            continue
                    else:
                        # Numpy array observation (fallback)
                        obs = self.racechat.obs
                        x, y, theta = obs[0], obs[1], obs[2]
                        vx, vy = obs[3], obs[4]
                        ax, ay = obs[6], obs[7] if len(obs) > 7 else 0.0
                    
                    # Compute speed magnitude
                    s_speed = np.sqrt(vx**2 + vy**2)
                    
                    # Store trajectory
                    trajectory['s_pos'].append(x)
                    trajectory['d_pos'].append(y)
                    trajectory['s_speed'].append(s_speed)
                    trajectory['d_speed'].append(vy)
                    trajectory['ax'].append(ax)
                    trajectory['ay'].append(ay)
                    trajectory['time'].append(time.time() - start_time)
                except (KeyError, IndexError, TypeError) as e:
                    # If observation format is unexpected, skip this step
                    if debug and len(trajectory['time']) == 0:
                        print(f"[DEBUG] Observation parsing error: {type(e).__name__}: {e}")
                    continue
            else:
                if debug and len(trajectory['time']) == 0:
                    print(f"[DEBUG] WARNING: obs is None!")
            
            # Step simulation with LLM-selected MPC parameters
            steer = self.racechat._mpc_params_to_steering(mpc_params)
            speed = float(np.clip(
                mpc_params.get('v_max', 12.0) * 0.4,
                mpc_params.get('v_min', 1.0),
                mpc_params.get('v_max', 12.0),
            ))
            action = np.array([[steer, speed]])
            try:
                self.racechat.obs, reward, done, truncated, info = self.racechat.env.step(action)
                if done or truncated:
                    break
            except Exception as e:
                print(f"Simulation step error: {e}")
                break
        
        return trajectory
    
    def test(self, test_case, prompt, ref_val=None):
        """
        Run a single test case.
        
        Args:
            test_case: 'center', 'refvel', 'reverse', or 'smooth'
            prompt: Natural language prompt for the task
            ref_val: Reference value for refvel tests
            
        Returns:
            Dictionary with test result
        """
        print(f"\n{'='*60}")
        print(f"TEST CASE: {test_case} | MEMORY: 5")
        print(f"{'='*60}")
        
        self.reset_mpc_params()
        
        # Run simulation episode
        episode = self.run_sim_episode(prompt=prompt)
        
        # Calculate RMSE based on trajectory
        trajectory = episode['trajectory']
        rmse = 69.0  # Default high error
        
        if test_case == "center":
            rmse = self.center_rmse(trajectory)
        elif test_case == "refvel":
            rmse = self.refvel_rmse(trajectory, target_speed=ref_val)
        elif test_case == "reverse":
            rmse = self.reverse_rmse(trajectory, target_speed=self.rev_target_speed)
        elif test_case == "smooth":
            rmse = self.smooth_rmse(trajectory)
        
        result = {
            'mem_nb': 5,
            'llm_cmd': episode['command'],
            'llm_expl': episode['explanation'],
            'llm_inf_latency': episode['llm_inf_latency'],
            'rmse': rmse,
            'case': test_case,
            'mem_sources': episode['mem_sources'],
            'trajectory': trajectory
        }
        
        print(f"RMSE: {rmse:.3f} m")
        print(f"LLM Latency: {episode['llm_inf_latency']:.2f} s")
        
        return result
    
    def get_default_baseline(self, test_case, ref_val=None):
        """
        Get baseline RMSE with default MPC parameters (no LLM tuning).
        
        Args:
            test_case: 'center', 'refvel', 'reverse', or 'smooth'
            ref_val: Reference value for refvel tests
            
        Returns:
            Baseline RMSE value
        """
        print(f"\nGetting baseline for {test_case}...")
        
        self.reset_mpc_params()
        trajectory = self.collect_trajectory(timeout=30, debug=self.debug_enabled)
        
        rmse = 69.0
        if test_case == "center":
            rmse = self.center_rmse(trajectory)
        elif test_case == "refvel":
            rmse = self.refvel_rmse(trajectory, target_speed=ref_val)
        elif test_case == "reverse":
            rmse = self.reverse_rmse(trajectory, target_speed=self.rev_target_speed)
        elif test_case == "smooth":
            rmse = self.smooth_rmse(trajectory)
        
        print(f"Baseline {test_case} RMSE: {rmse:.3f} m")
        return rmse
    
    # ========================================================================
    # RMSE Calculation Functions
    # ========================================================================
    
    def center_rmse(self, trajectory):
        """
        Calculate RMSE for centerline tracking.
        Lower is better (0 = perfect centerline).
        """
        if not trajectory['d_pos'] or len(trajectory['d_pos']) == 0:
            return 69.69
        
        d_poses = np.array(trajectory['d_pos'])
        # RMSE is absolute deviation from center (d_pos = 0)
        rmse = np.sqrt(np.mean(d_poses ** 2))
        return float(rmse)
    
    def refvel_rmse(self, trajectory, target_speed):
        """
        Calculate RMSE for reference velocity tracking.
        Lower is better (0 = perfect speed tracking).
        """
        if not trajectory['s_speed'] or len(trajectory['s_speed']) == 0:
            return 69.69
        
        s_speeds = np.array(trajectory['s_speed'])
        rmse = np.sqrt(np.mean((s_speeds - target_speed) ** 2))
        return float(rmse)
    
    def reverse_rmse(self, trajectory, target_speed):
        """
        Calculate RMSE for reverse driving.
        Lower is better (target_speed = -1.0 m/s).
        """
        if not trajectory['s_speed'] or len(trajectory['s_speed']) == 0:
            return 69.69
        
        s_speeds = np.array(trajectory['s_speed'])
        rmse = np.sqrt(np.mean((s_speeds - target_speed) ** 2))
        return float(rmse)
    
    def smooth_rmse(self, trajectory):
        """
        Calculate RMSE for smooth driving (low acceleration).
        Lower is better (0 = no acceleration).
        """
        if not trajectory['ax'] or len(trajectory['ax']) == 0:
            return 69.69
        
        ax = np.array(trajectory['ax'])
        ay = np.array(trajectory['ay'])
        combined_accel = np.sqrt(ax ** 2 + ay ** 2)
        rmse = np.sqrt(np.mean(combined_accel ** 2))
        return float(rmse)
    
    # ========================================================================
    # Test Running and Reporting
    # ========================================================================
    
    def run_tests(self, num_memories=5):
        """
        Run all test cases with LLM parameter tuning.
        
        Args:
            num_memories: Number of memory examples to use in RAG
            
        Returns:
            Dictionary with all test results and baselines
        """
        all_reports = {}
        
        for test_case in TEST_OPTIONS:
            print(f"\n\n{'#'*70}")
            print(f"# TESTING: {test_case.upper()}")
            print(f"{'#'*70}")
            
            all_reports[test_case] = {"subcases": {}}
            
            # Get test cases for this category
            test_prompts = EVAL_DRIVING_CASES.get(test_case, [])
            
            if not test_prompts:
                print(f"No test cases found for {test_case}")
                continue
            
            per_case_rmses = np.zeros(len(test_prompts))
            default_rmses = np.zeros(len(test_prompts))
            
            for idx, prompt_dict in enumerate(test_prompts):
                print(f"\n[{idx+1}/{len(test_prompts)}] {test_case}")
                
                # Extract prompt and reference value if needed
                prompt = prompt_dict['human_prompt']
                ref_val = prompt_dict.get('ref_val', None)
                baseline_rmse = prompt_dict.get('baseline_rmse', 69.0)
                
                # Get baseline (default params)
                default_rmse = self.get_default_baseline(test_case, ref_val=ref_val)
                default_rmses[idx] = default_rmse
                
                # Run test with LLM tuning
                result = self.test(test_case=test_case, prompt=prompt, ref_val=ref_val)
                per_case_rmses[idx] = result['rmse']
                
                # Store result
                all_reports[test_case]["subcases"][str(idx)] = {
                    "result": result,
                    "default_rmse": default_rmse,
                    "baseline_rmse": baseline_rmse
                }
            
            # Calculate aggregate statistics
            valid_rmses = per_case_rmses[per_case_rmses != 69.0]
            valid_defaults = default_rmses[default_rmses != 69.0]
            
            all_reports[test_case]["avg_rmse"] = float(np.mean(valid_rmses)) if len(valid_rmses) > 0 else 69.0
            all_reports[test_case]["std_rmse"] = float(np.std(valid_rmses)) if len(valid_rmses) > 0 else 0.0
            all_reports[test_case]["avg_default_rmse"] = float(np.mean(valid_defaults)) if len(valid_defaults) > 0 else 69.0
            all_reports[test_case]["std_default_rmse"] = float(np.std(valid_defaults)) if len(valid_defaults) > 0 else 0.0
            
            if all_reports[test_case]["avg_default_rmse"] > 0:
                improvement = (all_reports[test_case]["avg_default_rmse"] - all_reports[test_case]["avg_rmse"]) / all_reports[test_case]["avg_default_rmse"]
            else:
                improvement = 0.0
            all_reports[test_case]["avg_rmse_improvement"] = improvement
        
        return all_reports
    
    def generate_summary_report(self, reports):
        """
        Generate markdown summary report of all test results.
        
        Args:
            reports: Dictionary of test reports
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(self.eval_dir, f"_summary_report_{timestamp}.md")
        
        with open(report_path, 'w') as f:
            f.write(f"# Evaluation Summary - {self.model_name}\n\n")
            f.write(f"**Run Name:** {self.run_name}\n")
            f.write(f"**Date:** {timestamp}\n")
            f.write(f"**Model Dir:** {self.racechat.env}\n\n")
            f.write("## Test Case Results\n\n")
            
            for test_case, case_data in reports.items():
                avg_rmse = case_data["avg_rmse"]
                std_rmse = case_data["std_rmse"]
                avg_default = case_data["avg_default_rmse"]
                std_default = case_data["std_default_rmse"]
                improvement = case_data["avg_rmse_improvement"] * 100
                
                f.write(f"### {test_case.capitalize()}\n\n")
                f.write(f"- **LLM-Tuned RMSE**: {avg_rmse:.3f} ± {std_rmse:.3f} m\n")
                f.write(f"- **Default RMSE**: {avg_default:.3f} ± {std_default:.3f} m\n")
                f.write(f"- **Improvement**: {improvement:+.2f}%\n")
                f.write(f"- **Subcases:**\n")
                
                for idx, subcase in case_data["subcases"].items():
                    rmse = subcase["result"]["rmse"]
                    default_rmse = subcase["default_rmse"]
                    baseline = subcase["baseline_rmse"]
                    f.write(f"  - Subcase {idx}: RMSE={rmse:.3f}m (Default={default_rmse:.3f}m, Baseline={baseline:.3f}m)\n")
                
                f.write("\n")
        
        print(f"\n{'='*70}")
        print(f"Summary report saved to: {report_path}")
        print(f"{'='*70}")
        return report_path


def main():
    parser = argparse.ArgumentParser(description='Run simulation-based MPC tester.')
    parser.add_argument('--model', choices=['gpt-4o', 'custom', 'training'], 
                        default='custom', help='Model type to test')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to model directory')
    parser.add_argument('--quant', action='store_true',
                        help='Use 4-bit quantization')
    parser.add_argument('--render', action='store_true',
                        help='Enable visualization (requires display; disabled by default for headless systems like HPRC)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging to diagnose observation format issues')
    parser.add_argument('--host_ip', type=str, default='127.0.0.1',
                        help='Host IP (unused for simulation, kept for compatibility)')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv(find_dotenv())
    
    # Create run name
    run_name = os.path.basename(args.model_dir)
    run_name = f"{run_name}_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n{'='*70}")
    print(f"SIMULATION-BASED MPC PARAMETER TESTER")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Model Dir: {args.model_dir}")
    print(f"Quantization: {args.quant}")
    print(f"Render: {args.render}")
    print(f"Debug: {args.debug}")
    print(f"Run Name: {run_name}")
    print(f"{'='*70}\n")
    
    # Create tester
    tester = SimTester(
        model_type=args.model,
        model_dir=args.model_dir,
        quant=args.quant,
        run_name=run_name,
        host_ip=args.host_ip,
        render=args.render,
        debug=args.debug
    )
    
    # Run all tests
    reports = tester.run_tests(num_memories=5)
    
    # Generate summary
    tester.generate_summary_report(reports)
    
    print(f"\n{'='*70}")
    print("Test complete!")
    print(f"Results saved to: {tester.eval_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
