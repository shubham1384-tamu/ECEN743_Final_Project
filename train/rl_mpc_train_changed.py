import os, torch, yaml, argparse, time
num_gpus = torch.cuda.device_count()
os.environ["CUDA_VISIBLE_DEVICES"] = f"{num_gpus - 1}"
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from datetime import datetime
from trl import GRPOTrainer, GRPOConfig
from transformers import TrainerCallback
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
import numpy as np
import wandb
from dotenv import load_dotenv, find_dotenv
from functools import partial

# No roslibpy — gym-only, matches llm_mpc.py
from train.utils.mpc.mpc_dataset import MPCDatasetGRPO
import train.utils.mpc.eval_driving as eval_driving
from llm_mpc import RaceLLMMPC as RaceLLM

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", None)
MPC_PARAM_NAMES = ["qv", "qn", "qalpha", "qac", "qddelta", "alat_max",
                   "a_min", "a_max", "v_min", "v_max", "track_safety_margin"]


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def init_wandb():
    wandb_api_key = WANDB_API_KEY
    if not wandb_api_key:
        print("WANDB_API_KEY not found in environment variables.")
        print("Please provide your wandb API key or press Enter to skip:")
        user_input = input()
        if user_input.strip():
            wandb_api_key = user_input.strip()
        else:
            print("Skipping wandb logging.")
            return False
    try:
        wandb.login(key=wandb_api_key)
        print("Successfully logged in to Weights & Biases.")
        return True
    except Exception as e:
        print(f"Failed to log in to Weights & Biases: {e}")
        return False


def chat_mapping(chat_template="phi-3"):
    if chat_template == "phi-3":
        return {"role": "from", "content": "value", "user": "human", "assistant": "gpt"}
    elif chat_template == "qwen-2.5":
        return {"role": "role", "content": "content", "user": "user", "assistant": "assistant"}
    else:
        raise ValueError(f"Chat template {chat_template} not recognized.")


def train(cfg):
    out_dir        = cfg["training"]["out_dir"]
    chat_template  = cfg["training"]["chat_template"]
    use_rag        = cfg["training"]["use_rag"]
    base_model     = cfg["model"]["base_model"]
    load_in_4bit   = cfg["model"]["load_in_4bit"]
    train_bool     = cfg["training"]["train_bool"]
    lora_rank      = cfg["training"]["lora_rank"]
    wandb_project  = cfg["wandb"]["project"]
    max_seq_length = cfg["model"]["max_seq_length"]
    experiment_name = (base_model.split("/")[-1] + "_GRPO_" +
                       datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # ── F1TENTH config ────────────────────────────────────────────────────────
    f1tenth_map   = cfg["training"].get("f1tenth_map", "vegas")
    f1tenth_steps = cfg["training"].get("f1tenth_steps", 50)

    # ── Reward weights — set w2/w3/w4=0 to reproduce paper baseline ──────────
    w1 = cfg.get("reward", {}).get("w1", 1.0)   # Rtask
    w2 = cfg.get("reward", {}).get("w2", 0.0)   # Rstability
    w3 = cfg.get("reward", {}).get("w3", 0.0)   # Rsafety
    w4 = cfg.get("reward", {}).get("w4", 0.0)   # Refficiency
    w6 = cfg.get("reward", {}).get("w6", 1.0)   # Rfmt

    use_wandb = init_wandb()
    report_to = "wandb" if use_wandb else "none"

    if use_wandb:
        wandb.init(project=wandb_project, config=cfg)
    run_name = wandb.run.name if use_wandb else "no_wandb"

    # ── Model ─────────────────────────────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=cfg["model"]["fast_inference"],
        max_lora_rank=lora_rank,
        gpu_memory_utilization=cfg["model"]["gpu_memory_utilization"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=cfg["model"]["target_modules"],
        lora_alpha=cfg["model"]["lora_alpha"],
        use_gradient_checkpointing=cfg["model"]["use_gradient_checkpointing"],
        random_state=cfg["model"]["random_state"],
    )
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template,
        mapping=chat_mapping(),
    )

    # ── Environment — F1TENTH gym only, matches llm_mpc.py ───────────────────
    print(f"[INFO] Using F1TENTH gym backend (map={f1tenth_map}, steps={f1tenth_steps})")
    race_llm = RaceLLM(
        openai_token=OPENAI_API_TOKEN,
        model='training',
        f1tenth_map=f1tenth_map,
    )

    ##########################################################################
    # REWARD FUNCTIONS
    ##########################################################################

    step_state = {"step": 0}

    def wrapped_driving_reward(*args, **kwargs):
        kwargs["step"] = step_state["step"]
        return driving_adaption_reward(*args, **kwargs)

    def wrapped_format_reward(*args, **kwargs):
        kwargs["step"] = step_state["step"]
        return format_reward_func(*args, **kwargs)

    # ── Format reward ─────────────────────────────────────────────────────────
    def compute_format_score(completion, tag):
        open_tags  = completion.count(f"<{tag}>")
        close_tags = completion.count(f"</{tag}>")
        return 1.0 if open_tags == close_tags == 1 else 0.0

    def format_reward_func(prompts, completions, **kwargs) -> list[float]:
        reasoning_reward = np.zeros(len(prompts))
        answer_reward    = np.zeros(len(prompts))
        for i in range(len(prompts)):
            reasoning_reward[i] = compute_format_score(completions[i], "reasoning")
            answer_reward[i]    = compute_format_score(completions[i], "answer")
        reward = (reasoning_reward + answer_reward) / 2.0 * w6
        print(f"format_reward_func: {reward}")
        return [reward]

    # ── Param extraction reward ───────────────────────────────────────────────
    def mpc_param_extraction_reward(prompts, completions, evaluation_function,
                                    baseline_rmse, **kwargs) -> list[float]:
        rewards = np.zeros(len(prompts))
        for i in range(len(prompts)):
            extracted_command, _ = race_llm._sanitize_tune_output(completions[i])
            rewards[i] = 0.0 if extracted_command is None else 1.0
        print(f"mpc_param_extraction_reward = {rewards}")
        return [rewards]

    # ── Param name reward ─────────────────────────────────────────────────────
    def mpc_param_name_reward(prompts, completions, evaluation_function,
                               baseline_rmse, **kwargs) -> list[float]:
        rewards = np.zeros(len(prompts))
        for i in range(len(prompts)):
            extracted_command, _ = race_llm._sanitize_tune_output(completions[i])
            if extracted_command is None:
                rewards[i] = 0.0
                continue
            correct_params = 0
            partially_correct_params = 0
            if len(extracted_command) == 0:
                rewards[i] = 1.0
            else:
                for predicted_param_name in extracted_command.keys():
                    if not isinstance(predicted_param_name, str):
                        continue
                    if predicted_param_name in MPC_PARAM_NAMES:
                        correct_params += 1
                    elif (any(predicted_param_name in name for name in MPC_PARAM_NAMES) or
                          _parse_mpc_param_name(predicted_param_name) is not None):
                        partially_correct_params += 1
                rewards[i] = ((correct_params + 0.5 * partially_correct_params) /
                              len(extracted_command.keys()))
        print(f"mpc_param_name_reward = {rewards}")
        return [rewards]

    # # ── New reward term helpers ───────────────────────────────────────────────

    # def _stability_reward(param_history: list) -> float:
    #     """Rstability: penalizes oscillatory MPC params. No sim needed."""
    #     if len(param_history) < 3:
    #         return 0.0
    #     recent    = param_history[-5:]
    #     key_params = ['qv', 'qn', 'qalpha']
    #     variances = [np.var([h.get(k, 0.0) for h in recent]) for k in key_params]
    #     return float(np.exp(-2.0 * np.mean(variances)))

    # def _safety_reward(scan_minimums: list) -> float:
    #     """Rsafety: minimum lidar readings from gym rollout."""
    #     if not scan_minimums:
    #         return 0.0
    #     min_dist = float(np.min(scan_minimums))
    #     if min_dist < 0.2:
    #         return -1.0
    #     elif min_dist < 0.38:
    #         return -0.5
    #     return min(1.0, min_dist / 2.0)

    # def _efficiency_reward(params: dict, mean_speed: float) -> float:
    #     """Refficiency: penalizes high control effort and speed error."""
    #     effort_keys  = ['qn', 'qalpha', 'qac', 'qddelta']
    #     effort       = np.mean([abs(params.get(k, 0)) for k in effort_keys])
    #     effort_score = float(np.exp(-effort / 20.0))
    #     v_max        = params.get('v_max', 12.0)
    #     speed_score  = float(np.exp(-abs(mean_speed - v_max * 0.5) / 5.0))
    #     return 0.5 * effort_score + 0.5 * speed_score

    # ── Main driving reward — gym only ────────────────────────────────────────
    def driving_adaption_reward(prompts, completions, evaluation_function,
                                baseline_rmse, **kwargs) -> list[float]:
        print("###########Driving Reward###########")
        rewards = np.zeros(len(prompts))
        rmses   = np.zeros(len(prompts))

        for i in range(len(prompts)):
            # Reset sim between rollouts
            race_llm.reset_f1tenth()

            extracted_command, _ = race_llm._sanitize_tune_output(completions[i])
            if extracted_command is None:
                rmses[i] = 69.0
                continue

            # Sanitize param names
            sanitized = extracted_command.copy()
            for param in list(sanitized.keys()):
                if param not in MPC_PARAM_NAMES:
                    parsed = _parse_mpc_param_name(param)
                    if parsed:
                        sanitized[parsed] = sanitized.pop(param)
                    else:
                        sanitized.pop(param)

            full_params = {**race_llm.DEFAULT_MPC_PARAMS, **sanitized}

            # Run sim for N steps
            trajectory_d  = []   # lateral deviation → RMSE equivalent
            trajectory_v  = []   # speed → efficiency term
            scan_minimums = []   # min lidar → safety term

            for _ in range(f1tenth_steps):
                obs, done = race_llm._step_f1tenth(full_params)
                odom = race_llm._get_f1tenth_odom()

                trajectory_d.append(abs(odom['d_pos'][-1]))
                trajectory_v.append(abs(odom['s_speed'][-1]))

                if obs is not None and 'scans' in obs:
                    scan_minimums.append(float(np.min(obs['scans'][0])))

                if done:
                    break

            crashed = race_llm.f1tenth_done

            if crashed:
                rmses[i] = 69.0
                race_llm.reset_f1tenth()
            else:
                rmses[i] = float(np.mean(trajectory_d))

            # New reward terms
            # r_stability  = _stability_reward(race_llm.param_history)
            # r_safety     = _safety_reward(scan_minimums)
            # r_efficiency = _efficiency_reward(
            #     full_params,
            #     float(np.mean(trajectory_v)) if trajectory_v else 0.0,
            # )
            # extra = w2 * r_stability + w3 * r_safety + w4 * r_efficiency
            # rewards[i] = extra   # w1 * Rtask added below

            if use_wandb:
                wandb.log({
                    "reward/r_stability":  r_stability,
                    "reward/r_safety":     r_safety,
                    "reward/r_efficiency": r_efficiency,
                    "reward/extra":        extra,
                })

        # w1 * Rtask
        rewards += w1 * _rmse_to_reward(rmses, baseline_rmse)

        if use_wandb:
            token_lengths = [len(tokenizer.tokenize(c)) for c in completions]
            wandb.log({"output_token_lengths": {
                "avg": np.mean(token_lengths),
                "min": np.min(token_lengths),
                "max": np.max(token_lengths),
            }})

        print(f"Driving reward: {rewards}, RMSEs: {rmses}")
        print("###########Driving Reward End###########")
        return [rewards]

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _parse_mpc_param_name(predicted_param_name: str) -> str:
        param_mapping = {
            "qv":                  ["q_v", "weight_qv", "param_qv", "weight_q_v", "param_q_v"],
            "qn":                  ["q_n", "weight_qn", "param_qn", "weight_q_n", "param_q_n"],
            "qalpha":              ["q_alpha", "weight_qalpha", "param_qalpha", "weight_q_alpha", "param_q_alpha"],
            "qac":                 ["q_ac", "weight_qac", "param_qac", "weight_q_ac", "param_q_ac"],
            "qddelta":             ["q_ddelta", "weight_qddelta", "param_qddelta", "weight_q_ddelta", "param_q_ddelta"],
            "alat_max":            ["a_lat_max", "weight_alat_max", "param_alat_max", "weight_a_lat_max", "param_a_lat_max"],
            "a_min":               ["a_min", "weight_a_min", "param_a_min"],
            "a_max":               ["a_max", "weight_a_max", "param_a_max"],
            "v_min":               ["v_min", "weight_v_min", "param_v_min"],
            "v_max":               ["v_max", "weight_v_max", "param_v_max"],
            "track_safety_margin": ["track_safety_margin", "weight_track_safety_margin",
                                    "param_track_safety_margin"],
        }
        for standard_name, variations in param_mapping.items():
            if predicted_param_name in variations:
                return standard_name
        return None

    def _rmse_to_reward(rmses: np.ndarray, baseline_rmse: float,
                        min_out: float = 0.0) -> np.ndarray:
        if rmses.size == 0:
            return np.ones_like(rmses, dtype=float) * min_out
        rel_improvement = (baseline_rmse - rmses) / baseline_rmse
        rel_improvement = np.clip(rel_improvement, -4, 4)
        return rel_improvement

    class WandbCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and use_wandb:
                wandb.log(logs)

    class StepTrackerCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            step_state["step"] = state.global_step

    ##########################################################################
    # DATASET
    ##########################################################################

    dataset = MPCDatasetGRPO(
        test_cases=eval_driving.DRIVING_CASES,
        use_rag=use_rag,
        index=race_llm.decision_index,
        mem_nb=cfg["dataset"]["mem_nb"],
        shuffle=cfg["dataset"]["shuffle"],
    )
    print(f"Training MPCxLLM with {len(dataset)} samples.")

    ##########################################################################
    # TRAINING
    ##########################################################################

    if train_bool:
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
            max_grad_norm=0.1,
            report_to=report_to,
            output_dir=f"{out_dir}/{experiment_name}",
        )

        callbacks = [WandbCallback(), StepTrackerCallback()]

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[
                wrapped_driving_reward,
                wrapped_format_reward,
                mpc_param_extraction_reward,
                mpc_param_name_reward,
            ],
            args=training_args,
            train_dataset=dataset,
            callbacks=callbacks,
        )

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        if use_wandb:
            wandb.log({"gpu_info": {
                "name": gpu_stats.name,
                "max_memory_gb": max_memory,
                "reserved_memory_gb": start_gpu_memory,
            }})

        trainer_stats = trainer.train()

        if use_wandb:
            wandb.log({"training_complete": True, "training_stats": trainer_stats})

    model.save_pretrained(f"{out_dir}/{experiment_name}")
    print("Training complete, saved model: ", f"{out_dir}/{experiment_name}")

    if use_wandb:
        wandb.finish()

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False,
                        default="train/config/rl_mpc_train.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()
    train(load_config(args.config))