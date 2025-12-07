import unsloth
from unsloth import FastLanguageModel
import torch
import pandas as pd
from tqdm import tqdm

search_template = """Dataset Overview:
    - Please keep the exact initial description of the dataset as shown in beginning the prompt.

    Key Themes or Topics:
    - Central focus on a broad area of interest (e.g., urban planning, socio-economic factors, environmental analysis).
    - Data spans multiple subtopics or related areas that contribute to a holistic understanding of the primary theme.
    Example:
    - theme1/topic1
    - theme2/topic2
    - theme3/topic3

    Applications and Use Cases:
    - Facilitates analysis for professionals, policymakers, researchers, or stakeholders.
    - Useful for specific applications, such as planning, engineering, policy formulation, or statistical modeling.
    - Enables insights into patterns, trends, and relationships relevant to the domain.
    Example:
    - application1/usecase1
    - application2/usecase2
    - application3/usecase3

    Concepts and Synonyms:
    - Includes related concepts, terms, and variations to ensure comprehensive coverage of the topic.
    - Synonyms and alternative phrases improve searchability and retrieval effectiveness.
    Example:
    - concept1/synonym1
    - concept2/synonym2
    - concept3/synonym3

    Keywords and Themes:
    - Lists relevant keywords and themes for indexing, categorization, and enhancing discoverability.
    - Keywords reflect the dataset's content, scope, and relevance to the domain.
    Example:
    - keyword1
    - keyword2
    - keyword3

    Additional Context:
    - Highlights the dataset's relevance to specific challenges or questions in the domain.
    - May emphasize its value for interdisciplinary applications or integration with related datasets.
    Example:
    - context1
    - context2
    - context3"""


def build_prompt(dataset_info, user_focused=True):
    dataset_sample = dataset_info.get("data_example")
    description = dataset_info.get("description")
    title = dataset_info.get("dataset_name")
    agency = dataset_info.get("agency")
    category = dataset_info.get("category")
    column_definitions = dataset_info.get("column_info")
    tags = dataset_info.get("tags")


    introduction = f"""Answer the question using the following information."""

    dataset_sample_text = f"""First, consider the dataset sample:
    {dataset_sample}""" if dataset_sample is not None else ""

    initial_description = f"""The initial description is {description}.""" if description else ""
    title_agency_cat = f"""Additionally the dataset title is {title}, the agency is {agency} and the category is
    {category}. Based on this topic and agency, please add sentence(s) describing what this
    ataset can be used for.""" if title or agency or category else ""

    tag = f"""The tags are {tags}.""" if tags else ""

    column_defs = f"""Additionally, the column definitions are {column_definitions}.""" if column_definitions else ""

    user_closing = """Question: Based on the information above and the requirements, provide a dataset description in sentences. Use only natural,
    readable sentences without special formatting."""

    search_closing = """Please expand the initial description. Additionally, add as many related concepts, synonyms, and relevant terms as possible based on the initial description and the topic. Unlike the initial description, 
      which is focused on presentation and readability, the expanded description is intended to be indexed at backend of a dataset search engine to improve searchability. 
      Therefore, focus less on readability and more on including all relevant terms related to the topic. Make sure to include any variations of the key terms and concepts that 
      could help improve retrieval in search results. Please follow the structure of following example template:"""

    if user_focused:
        prompt = introduction + "\n" + dataset_sample_text + '\n' + initial_description + "\n" + \
                 title_agency_cat + "\n" + tag + "\n" + column_defs + "\n" + user_closing
    else:
        prompt = dataset_sample_text + '\n' + initial_description + "\n" + \
                 title_agency_cat + "\n" + tag + "\n" + column_defs + "\n" + search_closing + '\n' + search_template

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

    def generate_description(self, dataset_info, user_focused=True, temperature=0.0):
        """Generates a description given a prompt and temperature"""
        if user_focused:
            system_message = """You are an assistant for a dataset search engine. Your goal
    is to improve the readability of dataset descriptions for dataset search engine users."""
        else:
            system_message = """You are an assistant for a dataset search engine. Your goal is 
            to improve the performance of the dataset search engine for keyword queries."""

        user_content = build_prompt(dataset_info, user_focused=user_focused)[:5000]

        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
        )

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
        #text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        #return text.strip() #[len(prompt):].strip()

        gen_ids = outputs[0][inputs.input_ids.shape[-1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text.strip()


