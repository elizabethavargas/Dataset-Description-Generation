import unsloth
from unsloth import FastLanguageModel
import torch
import pandas as pd
from tqdm import tqdm
import re

# --- NEW: Constants for Smart Mode ---
FEW_SHOT_EXAMPLE = """
--- EXAMPLE OF A PERFECT DESCRIPTION ---
Input Data:
- Title: Restaurant Inspection Results
- Agency: Dept of Health
- Sample: [2020-01-01, 10A, Joe's Pizza, Vermin Violation]
- Columns: {date: Date of inspection, violation_code: Code for violation}

Output:
This dataset provides a record of sanitary violations in NYC restaurants, tracked by the Department of Health. It details individual inspections by 'Restaurant Name' and 'Date', including specific 'Violation Codes' (such as vermin or temperature issues). This data is useful for tracking public health compliance over time.
----------------------------------------
"""

CRITIC_SYSTEM_PROMPT = """You are a strict Data Quality Auditor.
Output your response in this exact format:
SCORE: [Number 1-5]
FEEDBACK: [One sentence explaining what is missing or wrong]

Criteria:
- If the description hallucinates columns not in the data, score 1-2.
- If it misses key columns, score 3.
- If it is accurate and comprehensive, score 5."""

REVISION_SYSTEM_PROMPT = """You are an Editor fixing a description based on a critique.
Rewrite the description to be 100% accurate to the SOURCE DATA and address the FEEDBACK.

IMPORTANT: Output ONLY the description. 
Do not add any conversational text like "Here is the rewritten version" or "Sure". 
Start directly with the description text."""


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

feedback_prompt_template = """
You generated the following dataset description:

{description}

The initial input used to generate this description was:

{instructions}

Your task: Critique the generated description for the following requirements: {requirements}.
- Point out missing information
- Highlight mistakes or inconsistencies
- Suggest what should be improved

Provide concise feedback that the model can use to revise its output.
"""

revision_prompt_template = """
You are now going to rewrite the dataset description.

Original instructions / input:
{instructions}

Feedback from previous critique:
{feedback}

Using the original input and the feedback, rewrite the dataset description to fully satisfy the requirements.
Provide only the updated dataset description in clear, natural sentences. Do not add extra commentary.
"""


def build_prompt(dataset_info, user_focused=True, include_example=False):
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
        prompt = introduction + "\n" 
        
        if include_example:
            prompt += FEW_SHOT_EXAMPLE + "\nNow, write a description for the following data:\n"
            
        prompt += dataset_sample_text + '\n' + initial_description + "\n" + \
                  title_agency_cat + "\n" + tag + "\n" + column_defs + "\n" + user_closing
        return prompt
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

    def _run_generation(self, user_content, system_message, temperature=0.0):
      """
      Runs a single-pass generation.
      - user_content: the main user prompt text
      - system_message: the system role content
      """

      prompt = self.tokenizer.apply_chat_template(
          [
              {"role": "system", "content": system_message},
              {"role": "user", "content": user_content},
          ],
          tokenize=False,
          add_generation_prompt=True,
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

      gen_ids = outputs[0][inputs.input_ids.shape[-1]:]
      text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
      return text.strip()

    def _generate_fewshot(self, base_prompt, user_focused=True, temperature=0.0):
      """
      Performs 3-step self-correcting few-shot generation:
      1. Initial generation
      2. Critique
      3. Revision
      """
      # Step 1 — initial output
      if user_focused:
            system_message = """You are an assistant for a dataset search engine. Your goal
      is to improve the readability of dataset descriptions for dataset search engine users."""
      else:
          system_message = """You are an assistant for a dataset search engine. Your goal is 
          to improve the performance of the dataset search engine for keyword queries."""
      initial_output = self._run_generation(base_prompt, system_message=system_message, temperature=temperature)

      # Step 2 — critique / feedback
      req_text = "user-focused description" if user_focused else "search-focused description"
      feedback_prompt = feedback_prompt_template.format(
          description=initial_output,
          instructions=base_prompt,
          requirements=req_text
      )
      feedback = self._run_generation(feedback_prompt, system_message=system_message, temperature=0.0)

      # Step 3 — rewrite using feedback
      revision_prompt = revision_prompt_template.format(
          instructions=base_prompt,
          feedback=feedback
      )
      final_output = self._run_generation(revision_prompt, system_message=system_message, temperature=temperature)

      return final_output
    
    def _parse_score(self, text):
        """Helper to extract score for Smart Mode."""
        try:
            match = re.search(r'SCORE:\s*(\d)', text, re.IGNORECASE)
            if match: return int(match.group(1))
            return int(re.search(r'\d', text).group(0))
        except:
            return 3

    
    def _generate_smart_loop(self, prompt_text, temperature=0.7):
        """The new DDX logic: Loop -> Grade -> Retry."""
        best_desc = self._run_generation(prompt_text, system_message="You are an expert Data Archivist.", temperature=temperature)
        
        for attempt in range(3):
            critique_input = f"PROMPT DATA:\n{prompt_text[:1500]}...\n\nGENERATED DESCRIPTION:\n{best_desc}"
            critique_response = self._run_generation(critique_input, system_message=CRITIC_SYSTEM_PROMPT, temperature=0.0)
            score = self._parse_score(critique_response)
            
            print(f"   [Smart Mode] Attempt {attempt+1}: Score {score}/5")
            
            if score >= 4:
                return best_desc
            
            feedback = critique_response.split('\n', 1)[-1].strip()
            revision_prompt = f"ORIGINAL DATA:\n{prompt_text}\n\nOLD DRAFT:\n{best_desc}\n\nFEEDBACK:\n{feedback}\n\nTask: Rewrite description."
            best_desc = self._run_generation(revision_prompt, system_message=REVISION_SYSTEM_PROMPT, temperature=temperature)
            
        return best_desc


    def generate_description(self, dataset_info, user_focused=True, few_shot=False, use_smart_loop=False, temperature=0.7):
        """
        Master function.
        - if use_smart_loop=True: Uses your new DDX Notebook logic (Critic Loop).
        - if few_shot=True: Uses the original Repo logic (Linear 3-step).
        - else: Uses standard single-pass generation.
        """
        
        # 1. Build Prompt (Enable example only if using smart loop)
        prompt = build_prompt(dataset_info, user_focused=user_focused, include_example=use_smart_loop)[:10000]

        
        if use_smart_loop:
            return self._generate_smart_loop(prompt, temperature=temperature)

        
        if few_shot:
            return self._generate_fewshot(prompt, user_focused=user_focused, temperature=temperature)

        #
        if user_focused:
            system_message = "You are an assistant for a dataset search engine. Your goal is to improve the readability of dataset descriptions."
        else:
            system_message = "You are an assistant for a dataset search engine. Your goal is to improve keyword search performance."

        return self._run_generation(prompt[:5000], system_message=system_message, temperature=temperature)