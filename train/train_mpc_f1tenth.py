#!/usr/bin/env python3
"""
LLM MPC Training on F1TENTH Gym Environment

This script trains an LLM to generate MPC (Model Predictive Control) parameters
for autonomous racing using the F1TENTH gym simulator.

Usage:
    python train_mpc_f1tenth.py --config train/config/rl_mpc_train_documented.yaml

Features:
    - GRPO (Group Relative Policy Optimization) training
    - LoRA-based efficient fine-tuning
    - Multi-reward functions (format, extraction, names, driving performance)
    - Weights & Biases integration for experiment tracking
    - F1TENTH gym environment integration
"""

import os
import sys

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================================================================
# Standard Imports (NOW safe to import torch with correct typing_extensions)
# ============================================================================
import torch
import yaml
import argparse
import time
import numpy as np
from datetime import datetime
from functools import partial

# ============================================================================
# ML/Training Imports
# ============================================================================
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from trl import GRPOTrainer, GRPOConfig
from transformers import TrainerCallback
import wandb
from dotenv import load_dotenv, find_dotenv

# ============================================================================
# Project Imports
# ============================================================================
from train.utils.mpc.mpc_dataset import MPCDatasetGRPO
import train.utils.mpc.eval_driving as eval_driving
from llm_mpc import RaceLLMMPC as RaceLLM

# ============================================================================
# Setup
# ============================================================================
# Configure GPU
num_gpus = torch.cuda.device_count()
os.environ["CUDA_VISIBLE_DEVICES"] = f"{num_gpus - 1}"

# Patch GRPO for Unsloth
PatchFastRL("GRPO", FastLanguageModel)

# Load environment variables
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", None)

# MPC Parameters to tune
MPC_PARAM_NAMES = [
    "qv", "qn", "qalpha", "qac", "qddelta", "alat_max",
    "a_min", "a_max", "v_min", "v_max", "track_safety_margin"
]


# ============================================================================
# Configuration & Utilities
# ============================================================================
def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def init_wandb() -> bool:
    """Initialize Weights & Biases for experiment tracking."""
    wandb_api_key = WANDB_API_KEY
    if not wandb_api_key:
        print("WANDB_API_KEY not found in environment variables.")
        print("Please provide your Weights & Biases API key or press Enter to skip:")
        user_input = input().strip()
        if user_input:
            wandb_api_key = user_input
        else:
            print("Skipping Weights & Biases logging.")
            return False
    
    try:
        wandb.login(key=wandb_api_key)
        print("✓ Successfully logged in to Weights & Biases.")
        return True
    except Exception as e:
        print(f"✗ Failed to log in to Weights & Biases: {e}")
        return False


def chat_mapping(chat_template: str = "qwen-2.5") -> dict:
    """Get chat template mapping for the model."""
    templates = {
        "phi-3": {"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        "qwen-2.5": {"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    }
    if chat_template not in templates:
        raise ValueError(f"Chat template '{chat_template}' not recognized. Use: {list(templates.keys())}")
    return templates[chat_template]


# ============================================================================
# Reward Functions
# ============================================================================
class RewardFunctions:
    """Collection of reward functions for GRPO training."""
    
    def __init__(self, race_llm, tokenizer, use_wandb=False, w1=1.0, w6=1.0):
        self.race_llm = race_llm
        self.tokenizer = tokenizer
        self.use_wandb = use_wandb
        self.w1 = w1
        self.w6 = w6
        self.step = 0

    def format_reward(self, prompts, completions, **kwargs) -> list:
        """Reward for proper XML formatting in model outputs."""
        reasoning_scores = np.zeros(len(prompts))
        answer_scores = np.zeros(len(prompts))
        
        for i, completion in enumerate(completions):
            reasoning_scores[i] = self._check_tags(completion, "reasoning")
            answer_scores[i] = self._check_tags(completion, "answer")
        
        reward = (reasoning_scores + answer_scores) / 2.0 * self.w6
        print(f"[Format Reward] Mean: {reward.mean():.3f}, Min: {reward.min():.3f}, Max: {reward.max():.3f}")
        return [reward]

    def extraction_reward(self, prompts, completions, **kwargs) -> list:
        """Reward for successfully extracting MPC parameters."""
        rewards = np.zeros(len(prompts))
        
        for i, completion in enumerate(completions):
            extracted, _ = self.race_llm._sanitize_tune_output(completion)
            rewards[i] = 1.0 if extracted is not None else 0.0
        
        print(f"[Extraction Reward] Success rate: {rewards.mean():.1%}")
        return [rewards]

    def param_name_reward(self, prompts, completions, **kwargs) -> list:
        """Reward for correctly named MPC parameters."""
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

    def driving_reward(self, prompts, completions, f1tenth_steps=50, f1tenth_map="vegas", 
                       baseline_rmse=2.0, **kwargs) -> list:
        """Reward based on F1TENTH gym driving performance."""
        print("=" * 60)
        print("[Driving Reward] Evaluating MPC parameters on F1TENTH gym...")
        print("=" * 60)
        
        # Handle baseline_rmse potentially being passed as list from trainer
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
            
            # Run simulation
            trajectory_d = []
            for step in range(f1tenth_steps):
                obs, done = self.race_llm._step_f1tenth(full_params)
                odom = self.race_llm._get_f1tenth_odom()
                trajectory_d.append(abs(odom['d_pos'][-1]))
                if done:
                    break
            
            # Compute RMSE and reward
            rmse = float(np.mean(trajectory_d)) if trajectory_d else 10.0
            rel_improvement = (baseline_rmse - rmse) / baseline_rmse
            rewards[i] = float(np.clip(rel_improvement, -4, 4))
            
            status = "✓" if rewards[i] > 0 else "✗"
            print(f"  Sample {i}: RMSE={rmse:.3f}, reward={rewards[i]:.3f} {status}")
            
            if self.race_llm.f1tenth_done:
                self.race_llm.reset_f1tenth()
        
        print(f"[Driving Reward] Mean: {rewards.mean():.3f}, Min: {rewards.min():.3f}")
        print("=" * 60)
        return [rewards * self.w1]

    @staticmethod
    def _check_tags(text: str, tag: str) -> float:
        """Check if XML tags are properly formatted."""
        open_count = text.count(f"<{tag}>")
        close_count = text.count(f"</{tag}>")
        return 1.0 if open_count == close_count == 1 else 0.0

    @staticmethod
    def _sanitize_params(extracted: dict) -> dict:
        """Sanitize parameter names and values."""
        sanitized = {}
        for param, value in extracted.items():
            if param in MPC_PARAM_NAMES:
                sanitized[param] = value
        return sanitized

    @staticmethod
    def _validate_params(params: dict) -> bool:
        """Validate that all parameters are non-None and numeric."""
        for key, value in params.items():
            if value is None:
                return False
            try:
                float(value)  # Ensure it's numeric
            except (TypeError, ValueError):
                return False
        return True


# ============================================================================
# Training
# ============================================================================
def train(config_path: str):
    """Main training function."""
    print("\n" + "=" * 70)
    print("LLM MPC Training on F1TENTH Gym Environment")
    print("=" * 70)
    
    cfg = load_config(config_path)
    print(f"[Config] Loaded from: {config_path}\n")
    
    # ── Extract configuration ─────────────────────────────────────────────
    out_dir = cfg["training"]["out_dir"]
    chat_template = cfg["training"]["chat_template"]
    use_rag = cfg["training"]["use_rag"]
    base_model = cfg["model"]["base_model"]
    load_in_4bit = cfg["model"]["load_in_4bit"]
    train_bool = cfg["training"]["train_bool"]
    lora_rank = cfg["training"]["lora_rank"]
    wandb_project = cfg["wandb"]["project"]
    max_seq_length = cfg["model"]["max_seq_length"]
    
    f1tenth_map = cfg["training"].get("f1tenth_map", "vegas")
    f1tenth_steps = cfg["training"].get("f1tenth_steps", 50)
    
    w1 = cfg.get("reward", {}).get("w1", 1.0)
    w6 = cfg.get("reward", {}).get("w6", 1.0)
    
    experiment_name = f"{base_model.split('/')[-1]}_GRPO_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    # ── Initialize Weights & Biases ───────────────────────────────────────
    use_wandb = init_wandb()
    report_to = "wandb" if use_wandb else "none"
    
    if use_wandb:
        wandb.init(project=wandb_project, config=cfg, name=experiment_name)
    
    print(f"[Training] Experiment: {experiment_name}\n")
    
    # ── Load model ────────────────────────────────────────────────────────
    print(f"[Model] Loading {base_model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=cfg["model"]["fast_inference"],
        max_lora_rank=lora_rank,
        gpu_memory_utilization=cfg["model"]["gpu_memory_utilization"],
        local_files_only=True,  # HPRC fix: use only locally cached models, no internet
    )
    print(f"[Model] ✓ Loaded\n")
    
    # ── Setup LoRA ────────────────────────────────────────────────────────
    print(f"[LoRA] Rank={lora_rank}, Alpha={cfg['model']['lora_alpha']}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=cfg["model"]["target_modules"],
        lora_alpha=cfg["model"]["lora_alpha"],
        use_gradient_checkpointing=cfg["model"]["use_gradient_checkpointing"],
        random_state=cfg["model"]["random_state"],
    )
    print(f"[LoRA] ✓ Applied\n")
    
    # ── Setup tokenizer ───────────────────────────────────────────────────
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template,
        mapping=chat_mapping(chat_template),
    )
    print(f"[Tokenizer] Chat template: {chat_template}\n")
    
    # ── Initialize F1TENTH gym environment ────────────────────────────────
    print(f"[F1TENTH] Map: {f1tenth_map}, Sim steps per update: {f1tenth_steps}")
    race_llm = RaceLLM(
        openai_token=OPENAI_API_TOKEN,
        model='training',
        f1tenth_map=f1tenth_map,
    )
    print(f"[F1TENTH] ✓ Environment initialized\n")
    
    # ── Setup reward functions ────────────────────────────────────────────
    reward_funcs_obj = RewardFunctions(
        race_llm=race_llm,
        tokenizer=tokenizer,
        use_wandb=use_wandb,
        w1=w1,
        w6=w6,
    )
    
    # ── Load dataset ──────────────────────────────────────────────────────
    print("[Dataset] Loading training data...")
    dataset = MPCDatasetGRPO(
        test_cases=eval_driving.DRIVING_CASES,
        use_rag=use_rag,
        index=race_llm.decision_index,
        mem_nb=cfg["dataset"]["mem_nb"],
        shuffle=cfg["dataset"]["shuffle"],
    )
    print(f"[Dataset] ✓ Loaded {len(dataset)} samples\n")
    
    # ── Training ──────────────────────────────────────────────────────────
    if train_bool:
        print("[Training] Starting GRPO training...")
        
        training_args = GRPOConfig(
            use_vllm=True,
            learning_rate=cfg["grpo"]["learning_rate"],
            adam_beta1=cfg["grpo"]["adam_beta1"],
            adam_beta2=cfg["grpo"]["adam_beta2"],
            weight_decay=cfg["grpo"]["weight_decay"],
            warmup_ratio=cfg["grpo"]["warmup_ratio"],
            lr_scheduler_type=cfg["grpo"]["lr_scheduler_type"],
            optim=cfg["grpo"]["optim"],
            logging_steps=cfg["grpo"]["logging_steps"],
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            per_device_train_batch_size=cfg["grpo"]["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["grpo"]["gradient_accumulation_steps"],
            num_generations=cfg["grpo"]["num_generations"],
            max_prompt_length=cfg["grpo"]["max_prompt_length"],
            max_completion_length=cfg["grpo"]["max_completion_length"],
            max_steps=cfg["grpo"]["max_steps"],
            save_steps=cfg["grpo"]["save_steps"],
            max_grad_norm=cfg["grpo"]["max_grad_norm"],
            report_to=report_to,
            output_dir=f"{out_dir}/{experiment_name}",
        )
        
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[
                reward_funcs_obj.driving_reward,
                reward_funcs_obj.format_reward,
                reward_funcs_obj.extraction_reward,
                reward_funcs_obj.param_name_reward,
            ],
            args=training_args,
            train_dataset=dataset,
        )
        
        # Print GPU info
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"[GPU] {gpu_stats.name}, Max memory: {max_memory} GB\n")
        
        # Train
        trainer_stats = trainer.train()
        print(f"\n[Training] ✓ Complete\n")
        
        if use_wandb:
            wandb.log({"training_complete": True, "trainer_stats": trainer_stats})
    
    # ── Save model ────────────────────────────────────────────────────────
    model.save_pretrained(f"{out_dir}/{experiment_name}")
    print(f"[Model] Saved to: {out_dir}/{experiment_name}\n")
    
    if use_wandb:
        wandb.finish()
    
    print("=" * 70)
    print("Training complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LLM for MPC parameter generation on F1TENTH gym",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_mpc_f1tenth.py --config train/config/rl_mpc_train_documented.yaml
  python train_mpc_f1tenth.py --config train/config/rl_mpc_train.yaml
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="train/config/rl_mpc_train_documented.yaml",
        help="Path to YAML configuration file (default: rl_mpc_train_documented.yaml)"
    )
    
    args = parser.parse_args()
    train(args.config)
