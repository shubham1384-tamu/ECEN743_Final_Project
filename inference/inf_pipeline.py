from transformers import Pipeline, BitsAndBytesConfig
import os, time
import numpy as np
import torch

# Try to import Unsloth (optimized for GPU), fall back to standard transformers for CPU
try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    UNSLOTH_AVAILABLE = True
except (NotImplementedError, ImportError) as e:
    print(f"[WARN] Unsloth not available ({type(e).__name__}): Using standard transformers instead.")
    print("       Note: Inference will be slower on CPU. For best performance, run on GPU (HPRC).")
    UNSLOTH_AVAILABLE = False
    # Provide fallback for CPU inference
    FastLanguageModel = None
    get_chat_template = None

CHAT_TEMPLATE_OPTIONS = ["phi-3", "qwen-2.5"]

class RaceLLMPipeline(Pipeline):
    def __init__(self, chat_template, model_dir=None, max_seq_length=2048, max_new_tokes=512, dtype=torch.float16, load_in_4bit=False, model=None, tokenizer=None):
        os.environ["TOKENIZERS_PARALLELISM"] = "false" # Avoids a warning
        self.chat_template = chat_template
        self.max_new_tokes = max_new_tokes

        # Load model and tokenizer via dir if not passed directly
        if model is None or tokenizer is None:
            model, tokenizer = self._load_model_from_dir(model_dir, load_in_4bit, dtype, chat_template, max_seq_length)
        else:
            print("Model and tokenizer passed directly, skipping loading from dir.")
            
        self.model = model
        self.tokenizer = tokenizer

        super().__init__(model=self.model, tokenizer=self.tokenizer)

    def _load_model_from_dir(self, model_dir, load_in_4bit, dtype, chat_template, max_seq_length):
        # Config for Quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=dtype,
        )
        
        # Use Unsloth if available (GPU), otherwise fall back to standard transformers (CPU)
        if UNSLOTH_AVAILABLE and FastLanguageModel is not None:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_dir,
                max_seq_length=max_seq_length,
                dtype=dtype,
                quantization_config=quantization_config,
            )
            FastLanguageModel.for_inference(model)
        else:
            # CPU fallback using standard transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import os
            
            print("Loading base model (CPU mode - may be slow)...")
            base_model_name = "Qwen/Qwen2.5-3B-Instruct"
            
            # Set cache directory to avoid re-downloading
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            print(f"  Cache dir: {cache_dir}")
            print(f"  First run will download ~6GB. Subsequent runs will use cache.")
            
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=quantization_config if load_in_4bit else None,
                torch_dtype=dtype,
                device_map="cpu",
                cache_dir=cache_dir
            )
            
            # Load LoRA adapter
            print(f"Loading LoRA adapter from {model_dir}...")
            model = PeftModel.from_pretrained(model, model_dir)
            model = model.merge_and_unload()
            
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        if chat_template == "phi-3":
            if UNSLOTH_AVAILABLE and get_chat_template is not None:
                tokenizer = get_chat_template(
                    tokenizer,
                    chat_template=chat_template,
                    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
                )
        elif chat_template == "qwen-2.5":
            if UNSLOTH_AVAILABLE and get_chat_template is not None:
                tokenizer = get_chat_template(
                    tokenizer,
                    chat_template="qwen-2.5",
                    mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
                )
            else:
                # Manual template setup for CPU mode
                tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + '<|im_end|>' }}{% endif %}{% endfor %}"
        else:
            raise ValueError(f"Chat template {chat_template} not recognized. Please use 'phi-3' or 'qwen-2.5'.")
        
        # Add pad token if it does not exist
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
        messages = [{"from": "human", "value": text}] if self.chat_template == "phi-3" else [{"role": "user", "content": text}]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        # Manually construct inputs
        inputs = {
            "input_ids": encoded,  # Direct tensor
            "attention_mask": (encoded != self.tokenizer.pad_token_id).long(),  # Manually create attention mask
        }

        return inputs

    def _forward(self, model_inputs, **forward_params):
        outputs = self.model.generate(
            input_ids=model_inputs["input_ids"], 
            attention_mask=model_inputs["attention_mask"],  # Explicit attention mask
            max_new_tokens=self.max_new_tokes, 
            use_cache=True,
            temperature=1.0,
            do_sample=False, # Greedy decoding
            top_k=20, # 20 Qwen recommended
            top_p=0.8, # 0.8 Qwen recommended
            repetition_penalty=1.0, # 1.0 = no penalty
            encoder_repetition_penalty=1.0, # 1.0 = no penalty
            length_penalty=-1.0, # < 0.0 encourages shorter sentences
        )
        return self.tokenizer.batch_decode(outputs)

    def postprocess(self, model_outputs, **postprocess_params):
        if self.chat_template == "phi-3":
            # Filter out the chat template, we want what is between <|assistant|> and <|end|>
            model_outputs = [output.split("<|assistant|>")[1].split("<|end|>")[0].strip() for output in model_outputs]
            return model_outputs[0], None, None
        elif self.chat_template == "qwen-2.5":
            # Filter out the chat template for Qwen-2.5, extract content between <|im_start|>assistant and <|im_end|>
            model_outputs = [
                output.split("<|im_start|>assistant")[1].split("<|im_end|>")[0].strip() 
                if "<|im_start|>assistant" in output and "<|im_end|>" in output
                else output  # Fallback to raw output if markers are missing
                for output in model_outputs
            ]
            return model_outputs[0], None, None
        else:
            raise ValueError(f"Chat template {self.chat_template} not recognized.")

# Loads Prompt with hints
def load_prompt(prompt_type) -> str:
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
        raise ValueError(f"Prompt type {prompt_type} not recognized. Please use 'reasoning' or 'synthetic'.")
