import unsloth
from unsloth import FastLanguageModel
import torch
import pandas as pd
from tqdm import tqdm


def build_prompt(dataset_info):
    dataset_sample = dataset_info.get("data_example")
    description = dataset_info.get("description")
    title = dataset_info.get("dataset_name")
    agency = dataset_info.get("agency")
    category = dataset_info.get("category")
    column_definitions = dataset_info.get("column_info")
    tags = dataset_info.get("tags")
    dataset_id = dataset_info.get("dataset_id")

    # Build prompt
    system_message = """You are an assistant for a dataset search engine. Your goal
is to improve the readability of dataset descriptions for dataset search engine users."""

    introduction = f"""Answer the question using the following information.

First, consider the dataset sample:

{dataset_sample}""" if dataset_sample is not None else ""

    initial_description = f"""The initial description is {description}.""" if description else ""

    title_agency_cat = f"""Additionally the dataset title is {title}, the agency is {agency} and the category is
{category}. Based on this topic and agency, please add sentence(s) describing what this
dataset can be used for.""" if title or agency or category else ""

    tag = f"""The tags are {tags}.""" if tags else ""

    column_defs = f"""Additionally, the column definitions are {column_definitions}.""" if column_definitions else ""

    closing_instruction = """Question: Based on the information above and the
requirements, provide a dataset description in sentences. Use only natural,
readable sentences without special formatting."""

    prompt = system_message + "\n" + introduction + "\n" + initial_description + "\n" + \
             title_agency_cat + "\n" + tag + "\n" + column_defs + "\n" + closing_instruction

    return prompt

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

    def generate_description(self, dataset_info, temperature=0.0):
        """Generates a description given a prompt and temperature"""
        prompt = build_prompt(dataset_info)[:5000]

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

