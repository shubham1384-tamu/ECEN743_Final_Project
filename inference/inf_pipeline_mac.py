"""
Mac/CPU-compatible inference pipeline wrapper.
Handles device placement (CPU vs GPU) gracefully.
Wraps inf_pipeline.RaceLLMPipeline for compatibility with systems without CUDA.
"""

import torch
from transformers import Pipeline, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# Check if CUDA is available
HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAS_CUDA else "cpu"

print(f"[INFO] Device: {DEVICE} (CUDA available: {HAS_CUDA})")

CHAT_TEMPLATE_OPTIONS = ["phi-3", "qwen-2.5"]


class RaceLLMPipelineCompat(Pipeline):
    """
    CPU/GPU compatible inference pipeline.
    Automatically detects device and places tensors correctly.
    """
    
    def __init__(self, chat_template, model_dir=None, max_seq_length=2048, max_new_tokes=512, dtype=torch.float16, load_in_4bit=False, model=None, tokenizer=None):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.chat_template = chat_template
        self.max_new_tokes = max_new_tokes
        self.device = DEVICE

        # Load model and tokenizer via dir if not passed directly
        if model is None or tokenizer is None:
            model, tokenizer = self._load_model_from_dir(model_dir, load_in_4bit, dtype, chat_template, max_seq_length)
        else:
            print("Model and tokenizer passed directly, skipping loading from dir.")
            
        self.model = model
        self.tokenizer = tokenizer

        super().__init__(model=self.model, tokenizer=self.tokenizer)

    def _load_model_from_dir(self, model_dir, load_in_4bit, dtype, chat_template, max_seq_length):
        """Load model and apply LoRA adapter with correct device handling."""
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=dtype,
        ) if load_in_4bit else None
        
        print(f"Loading base model on {DEVICE}...")
        base_model_name = "Qwen/Qwen2.5-3B-Instruct"
        
        # Set cache directory
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            torch_dtype=dtype if DEVICE == "cuda" else torch.float32,  # Use float32 on CPU
            device_map=DEVICE,
            cache_dir=cache_dir
        )
        
        # Load LoRA adapter
        print(f"Loading LoRA adapter from {model_dir}...")
        model = PeftModel.from_pretrained(model, model_dir)
        model = model.merge_and_unload()
        
        # Move to device
        model = model.to(DEVICE)
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Apply chat template
        if chat_template == "phi-3":
            tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'from' %}<|user|>\n{{ message['value'] }}<|end|>\n{% elif message['role'] == 'assistant' %}<|assistant|>\n{{ message['value'] }}<|end|>\n{% endif %}{% endfor %}"
        elif chat_template == "qwen-2.5":
            tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n<|im_start|>assistant\n{% elif message['role'] == 'assistant' %}{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}"
        else:
            raise ValueError(f"Chat template {chat_template} not recognized.")
        
        # Add pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
            model.resize_token_embeddings(len(tokenizer))
            
        return model, tokenizer

    def _sanitize_parameters(self, **pipeline_parameters):
        preprocess_params = {}
        forward_params = {}
        postprocess_params = {}
        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, text, **preprocess_params):
        """Preprocess text - place on correct device."""
        messages = [{"role": "user", "content": text}]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)  # ← Use self.device instead of hardcoded "cuda"

        inputs = {
            "input_ids": encoded,
            "attention_mask": (encoded != self.tokenizer.pad_token_id).long(),
        }

        return inputs

    def _forward(self, model_inputs, **forward_params):
        """Generate output."""
        outputs = self.model.generate(
            input_ids=model_inputs["input_ids"], 
            attention_mask=model_inputs["attention_mask"],
            max_new_tokens=self.max_new_tokes, 
            use_cache=True,
            temperature=1.0,
            do_sample=False,
            top_k=20,
            top_p=0.8,
            repetition_penalty=1.0,
            encoder_repetition_penalty=1.0,
            length_penalty=-1.0,
        )
        return self.tokenizer.batch_decode(outputs)

    def postprocess(self, model_outputs, **postprocess_params):
        """Extract clean response from model output."""
        if self.chat_template == "phi-3":
            model_outputs = [output.split("<|assistant|>")[1].split("<|end|>")[0].strip() for output in model_outputs]
            return model_outputs[0], None, None
        elif self.chat_template == "qwen-2.5":
            model_outputs = [
                output.split("<|im_start|>assistant")[1].split("<|im_end|>")[0].strip() 
                if "<|im_start|>assistant" in output and "<|im_end|>" in output
                else output
                for output in model_outputs
            ]
            return model_outputs[0], None, None
        else:
            raise ValueError(f"Chat template {self.chat_template} not recognized.")


def load_prompt(prompt_type) -> str:
    """Load prompt hints from file."""
    if 'reasoning' in prompt_type:
        hints_dir = os.path.join('../', 'prompts/reasoning_hints.txt')
        with open(hints_dir, 'r') as f:
            reasoning_hints = f.read()
        return reasoning_hints
    elif 'synthetic' in prompt_type:
        hints_dir = os.path.join('../', 'prompts/example_synthetic.txt')
        with open(hints_dir, 'r') as f:
            synthetic_hints = f.read()
        return synthetic_hints
    else:
        raise ValueError(f"Prompt type {prompt_type} not recognized.")
