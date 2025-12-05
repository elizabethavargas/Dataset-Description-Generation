# pipeline.py

# Imports
import pandas as pd
import pickle
from generator.generator import HFGenerator  # <-- import your generator class

# Initialize your generators
qwen_generator = HFGenerator("unsloth/Qwen2-7B-Instruct")
llama_generator = HFGenerator("unsloth/Meta-Llama-3.1-8B-Instruct")

# Read datasets.pkl
datasets = pd.read_pickle("../datasets.pkl")

new_descriptions = {}

# Loop through datasets
for dataset in datasets:
    dataset_sample = dataset.get("data_example")
    description = dataset.get("description")
    title = dataset.get("dataset_name")
    agency = dataset.get("agency")
    category = dataset.get("category")
    column_definitions = dataset.get("column_info")
    tags = dataset.get("tags")
    dataset_id = dataset.get("dataset_id")

    # Build prompt
    system_message = f"""You are an assistant for a dataset search engine. Your goal
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

    # Generate descriptions
    qwen_description = qwen_generator.generate_description(prompt)
    llama_description = llama_generator.generate_description(prompt)

    # Save results
    new_descriptions[dataset_id] = {
        'qwen_description': qwen_description,
        'llama_description': llama_description
    }

# Save new_descriptions to pickle
with open('new_descriptions2.pkl', 'wb') as f:
    pickle.dump(new_descriptions, f)

print("Pipeline finished! Saved new_descriptions2.pkl")

