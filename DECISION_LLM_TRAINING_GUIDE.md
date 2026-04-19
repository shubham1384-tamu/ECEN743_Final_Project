# Decision LLM Training Guide with GRPO on F1tenth Simulator

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Prepare Training Data](#prepare-training-data)
3. [Configuration](#configuration)
4. [Start Training](#start-training)
5. [Monitor Training](#monitor-training)
6. [Training Test Cases](#training-test-cases)
7. [Quick Start](#quick-start)
8. [Troubleshooting](#troubleshooting)

---

## Environment Setup

### Install Dependencies
Ensure all required packages are installed:

```bash
pip install -r requirements.txt
```

**Key packages required:**
- `transformers>=4.36.0`
- `trl>=0.7.0` (for GRPO training)
- `unsloth` (for efficient training)
- `torch>=2.0.0`
- `wandb` (for experiment tracking)
- `peft` (for LoRA)

### Setup Environment Variables

Create or update your `.env` file in the project root:

```bash
# OpenAI API token (used for reward computation and RAG)
OPENAI_API_TOKEN=sk-your-token-here

# HuggingFace token (for model downloads)
HUGGINGFACEHUB_API_TOKEN=hf_your-token-here

# Weights & Biases API key (optional, for experiment tracking)
WANDB_API_KEY=your-wandb-key-here
```

Load the environment:
```bash
source .env
```

---

## Prepare Training Data

### Data Format

The training dataset requires robot state JSON files with the following structure:

```json
{
  "s_pos": [0.0, 0.1, 0.2, ...],           // Position along racing line (meters)
  "d_pos": [-0.05, 0.0, 0.05, ...],        // Lateral deviation from racing line (meters)
  "s_speed": [2.5, 2.6, 2.4, ...],         // Speed along racing line (m/s)
  "d_speed": [0.01, -0.02, 0.01, ...],     // Lateral speed (m/s)
  "d_left": [1.2, 1.15, 1.1, ...],         // Distance to left wall (meters)
  "d_right": [1.3, 1.35, 1.4, ...],        // Distance to right wall (meters)
  "theta": [0.5, 0.51, 0.49, ...],         // Heading angle (radians)
  "reversing": false,                       // Is car reversing?
  "data_samples": 10                        // Number of samples in this state
}
```

### Option A: Generate Synthetic Data (Recommended for Quick Start)

```bash
python train/data_generator/synthetic_data_generator.py
```

**Output:** Creates random robot states for quick testing
- Location: `train/dataset/synthetic_robot_data.json`
- Good for: Initial setup and debugging

### Option B: Collect Data from F1tenth Simulator

```bash
python train/data_generator/distil_reason_dataset_generator.py
```

**Requirements:**
- F1tenth simulator running
- ROS master active
- Connected to the simulator via network

**Output:** Real driving data from simulator
- Location: `train/dataset/` 
- Good for: Production training with realistic data

---

## Configuration

### Update Training Config File

Edit `train/config/rl_decision_train.yaml`:

```yaml
# === General Training Configuration ===
training:
  out_dir: "train/outputs"              # Output directory for checkpoints
  train_bool: true                      # Set to true to actually train
  use_rag: true                         # Use RAG for memory hints
  lora_rank: 64                         # LoRA rank (higher = more parameters)
  chat_template: "qwen-2.5"             # Chat template format

# === Weights & Biases ===
wandb:
  enabled: true                         # Enable experiment tracking
  project: "decision_grpo"              # W&B project name
  api_key: "${WANDB_API_KEY}"           # From environment

# === Model and Tokenizer ===
model:
  base_model: "Qwen/Qwen2.5-3B-Instruct"  # Base model to fine-tune
  load_in_4bit: true                      # 4-bit quantization (saves memory)
  max_seq_length: 2048                    # Maximum sequence length
  fast_inference: true                    # Use fast inference
  gpu_memory_utilization: 0.5             # GPU memory usage (0.0-1.0)
  lora_alpha: 64                          # LoRA alpha parameter
  use_gradient_checkpointing: "unsloth"   # Memory optimization
  random_state: 3407                      # Random seed

# === GRPO (Group Relative Policy Optimization) ===
grpo:
  use_vllm: true                          # Use vLLM for fast generation
  learning_rate: 0.000005                 # Initial learning rate
  adam_beta1: 0.9                         # Adam beta1
  adam_beta2: 0.99                        # Adam beta2
  weight_decay: 0.1                       # L2 regularization
  warmup_ratio: 0.1                       # Warmup ratio
  lr_scheduler_type: "cosine"             # Learning rate scheduler
  optim: "adamw_8bit"                     # Optimizer
  logging_steps: 1                        # Log every N steps
  per_device_train_batch_size: 1          # Batch size per GPU
  gradient_accumulation_steps: 1          # Gradient accumulation
  num_generations: 8                      # Generations per prompt
  max_prompt_length: 250                  # Max prompt tokens
  max_completion_length: 250              # Max response tokens
  max_steps: 800                          # Total training steps
  save_steps: 250                         # Save checkpoint every N steps
  max_grad_norm: 0.1                      # Gradient clipping

# === Dataset ===
dataset:
  raw_robot_states_dir: "/embodiedai/train/dataset/rl_decision/raw_states"
  mem_nb: 5                               # Number of RAG memory items
  shuffle: true                           # Shuffle dataset
  from_raw: false                         # Process raw states (if true)

# === Evaluation ===
evaluation:
  robot_states_dir: "/embodiedai/tests/decision_tester/robot_states"
  use_rag: true                           # Use RAG in evaluation
  eval_interval_steps: 400                # Evaluate every N steps

# === API Tokens ===
tokens:
  huggingfacehub: "${HUGGINGFACEHUB_API_TOKEN}"
  openai: "${OPENAI_API_TOKEN}"
  wandb: "${WANDB_API_KEY}"
```

### Key Configuration Parameters to Adjust

| Parameter | Impact | Adjustment |
|-----------|--------|------------|
| `per_device_train_batch_size` | Memory usage | ↓ if CUDA OOM |
| `num_generations` | Training diversity | ↑ for better training |
| `max_steps` | Training duration | ↑ for longer training |
| `learning_rate` | Training speed | ↓ if unstable |
| `lora_rank` | Model capacity | ↑ for more parameters |

---

## Start Training

### Run Training

```bash
python train/rl_decision_train.py --config train/config/rl_decision_train.yaml
```

### What Happens During Training

1. **Model Loading**
   - Loads Qwen2.5-3B-Instruct model
   - Applies 4-bit quantization for memory efficiency
   - Adds LoRA adapters for fine-tuning

2. **Dataset Loading**
   - Loads robot state JSON files
   - Creates prompts from 8 decision test cases
   - Combines each state with each test case

3. **Training Loop**
   - For each training step:
     - Generate 8 model responses per prompt (GRPO)
     - Compute correctness reward (does response match correct decision?)
     - Compute format reward (proper XML tags?)
     - Update model to maximize combined reward
     - Log metrics to Weights & Biases

4. **Checkpointing**
   - Saves model every `save_steps` steps
   - Evaluates on full test set every `eval_interval_steps` steps

### Expected Output

```
GPU = NVIDIA A100. Max memory = 80.0 GB.
15.234 GB of memory reserved.
Dataset created with 800 samples
Successfully logged in to Weights & Biases.
INFO - Training started...
Step 1/800: loss=2.345, correctness_reward=0.123, format_reward=0.456
Step 2/800: loss=2.212, correctness_reward=0.234, format_reward=0.567
...
```

---

## Monitor Training

### Weights & Biases Dashboard

If W&B is enabled, visit your project dashboard to see:
- Training loss curves
- Reward trends
- Sample model outputs
- Evaluation metrics

### Local Logs

Check console output for:
- Training loss
- Per-step reward values
- Sample model completions
- Evaluation results

### Checkpoints

Saved in: `train/outputs/Qwen2.5-3B_GRPO_<YYYY-MM-DD_HH-MM-SS>/`

```
train/outputs/
├── Qwen2.5-3B_GRPO_2024-04-19_14-32-10/
│   ├── adapter_model.bin         # LoRA weights
│   ├── adapter_config.json       # LoRA configuration
│   ├── training_args.bin         # Training configuration
│   └── trainer_state.json        # Training state
```

---

## Training Test Cases

The model is trained to handle 8 different decision scenarios:

### Test Case 1: Drive Forward
- **Human Prompt:** "Drive forward"
- **Expected Behavior:** s_speed > 0.1 m/s
- **Evaluation:** `forward_check()`
- **Success:** Car maintains positive speed

### Test Case 2: Stop Car
- **Human Prompt:** "Don't move"
- **Expected Behavior:** Both s_speed and d_speed ≈ 0
- **Evaluation:** `stop_car_check()`
- **Success:** Car comes to complete stop

### Test Case 3: Reverse
- **Human Prompt:** "Reverse the car"
- **Expected Behavior:** s_speed < 0 OR reversing flag = True
- **Evaluation:** `reverse_car_check()`
- **Success:** Car moves backward

### Test Case 4: Racing Line
- **Human Prompt:** "Drive on the racing line"
- **Expected Behavior:** |d_pos| ≤ 0.3m (centered on track)
- **Evaluation:** `drive_racing_line_check()`
- **Success:** Car stays within ±0.3m of racing line

### Test Case 5: Centerline
- **Human Prompt:** "Drive on the centerline"
- **Expected Behavior:** d_left ≈ d_right (equidistant from walls)
- **Evaluation:** `drive_centerline_check()`
- **Success:** Car drives down center of track

### Test Case 6: High Speed
- **Human Prompt:** "Drive faster than 3 m/s"
- **Expected Behavior:** s_speed > 3 m/s
- **Evaluation:** `speed_check()`
- **Success:** Car exceeds 3 m/s threshold

### Test Case 7: Left Wall
- **Human Prompt:** "Drive close to the left wall"
- **Expected Behavior:** d_left < 0.4m AND s_speed > 0.1
- **Evaluation:** `drive_close_to_left_wall_check()`
- **Success:** Car is within 0.4m of left wall

### Test Case 8: Oscillation
- **Human Prompt:** "Oscillate!"
- **Expected Behavior:** Multiple sign changes in d_pos AND max(|d_pos|) ≥ 0.3m
- **Evaluation:** `oscillation_check()`
- **Success:** Car oscillates side-to-side across track

---

## Quick Start

For fastest setup with minimal configuration:

```bash
# 1. Generate synthetic data
python train/data_generator/synthetic_data_generator.py

# 2. Set environment variables
export OPENAI_API_TOKEN="sk-..."
export HUGGINGFACEHUB_API_TOKEN="hf_..."

# 3. Run training with default config
python train/rl_decision_train.py
```

**This will:**
- Use default config from `train/config/rl_decision_train.yaml`
- Train for 800 steps
- Save checkpoints every 250 steps
- Evaluate every 400 steps

---

## Troubleshooting

### Issue: "No robot states found"
**Error Message:**
```
FileNotFoundError: [Errno 2] No such file or directory
```

**Solution:**
1. Check `raw_robot_states_dir` in config points to valid directory
2. Ensure directory contains `.json` files
3. Generate synthetic data: `python train/data_generator/synthetic_data_generator.py`
4. Verify JSON format matches expected structure

### Issue: CUDA Out of Memory (OOM)
**Error Message:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
1. Reduce `per_device_train_batch_size` (e.g., 1 → 1)
2. Reduce `num_generations` (e.g., 8 → 4)
3. Reduce `max_seq_length` (e.g., 2048 → 1024)
4. Enable gradient checkpointing (already enabled)

### Issue: OPENAI_API_TOKEN Not Found
**Error Message:**
```
ValueError: OPENAI_API_TOKEN not found in environment variables
```

**Solution:**
1. Add to `.env` file
2. Load environment: `source .env`
3. Verify: `echo $OPENAI_API_TOKEN`

### Issue: Training Too Slow
**Symptoms:**
- Steps take >30 seconds each
- GPU utilization <50%

**Solutions:**
1. Increase `num_generations` to 8+
2. Increase `per_device_train_batch_size` if VRAM allows
3. Use `use_vllm: true` in config
4. Check GPU is actually being used: `nvidia-smi`

### Issue: Model Not Improving
**Symptoms:**
- Loss doesn't decrease
- Rewards stay constant

**Solutions:**
1. Increase `learning_rate` (e.g., 5e-6 → 1e-5)
2. Check data quality (ensure correct labels in robot states)
3. Increase `max_steps` for longer training
4. Verify reward functions are working: check console output

### Issue: Model Output Format Wrong
**Symptoms:**
- Model generates text but not in expected format
- Format reward stays at 0

**Solutions:**
1. Ensure RAG memory has good examples
2. Check prompt format in `race_reasoning()` method
3. Reduce `max_completion_length` to force concise outputs
4. Manually check a model generation: compare with expected format

---

## Next Steps

1. **After Training:**
   - Evaluate model on held-out test set
   - Convert to inference format (GGUF or ONNX if needed)
   - Test on real F1tenth hardware or simulator

2. **For Better Results:**
   - Collect more diverse robot state data
   - Increase training steps (max_steps: 1000+)
   - Fine-tune learning rate and LoRA rank
   - Combine with MPC training for end-to-end learning

3. **Production Deployment:**
   - Quantize model (4-bit or 8-bit)
   - Use inference module: `inference/inf_pipeline.py`
   - Test on physical F1tenth car

---

## Support & References

- **Training Script:** `train/rl_decision_train.py`
- **Config File:** `train/config/rl_decision_train.yaml`
- **Dataset Class:** `train/utils/decision/decision_dataset.py`
- **Evaluation Functions:** `train/utils/decision/eval_states.py`
- **Main LLM Class:** `llm_mpc.py`

For issues or questions, check the repository documentation or reach out to the development team.

---

**Document Version:** 1.0  
**Last Updated:** 2026-04-19  
**Project:** ECEN 743 Final Project - F1tenth LLM-based Control
