import unsloth
from unsloth import FastLanguageModel
import torch
import pandas as pd
from tqdm import tqdm

generation_models = [
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    "unsloth/Qwen2-7B-Instruct",
]

class HFGenerator:
    """Generates descriptions using a Hugging Face model"""

    def __init__(self, model_name):
        if model_name not in generation_models:
            raise ValueError(f"Model '{model_name}' is not in the list of available models. "
                             f"Choose from: {generation_models}")

        self.model_name = model_name

        # Load model + tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
        )

        FastLanguageModel.for_inference(self.model)

        if "Qwen" in model_name:
            #self.tokenizer.pad_token = "<|extra_0|>"
            #self.tokenizer.eos_token = "</s>"
            #self.tokenizer.bos_token = "<s>"
            #self.eos_ids = [self.tokenizer.eos_token_id]

            self.tokenizer.eos_token = "<|im_end|>"       # real EOS
            self.tokenizer.pad_token = "<|endoftext|>"    # real PAD
            self.tokenizer.bos_token = self.tokenizer.pad_token
            self.eos_ids = [self.tokenizer.eos_token_id]
            self.pad_id = self.tokenizer.pad_token_id
            self.bos_id = self.tokenizer.bos_token_id

        else:  # LLaMA
            self.eos_ids = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

    def generate_description(self, prompt, temperature=0.0):
        """Generates a description given a prompt and temperature"""

        prompt = prompt[:5000]
	inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        do_sample = temperature > 0
    
        if "Llama" in self.model_name or "Meta-Llama" in self.model_name:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=do_sample,
                    temperature=temperature,
                    num_beams=1,
                    eos_token_id=self.eos_ids,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

        else:  # Qwen branch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=do_sample,
                    temperature=temperature,
                    eos_token_id=self.eos_ids,
                    pad_token_id=self.pad_id,
                    bos_token_id=self.bos_id,
                    use_cache=True,
                )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return text[len(prompt):].strip()

