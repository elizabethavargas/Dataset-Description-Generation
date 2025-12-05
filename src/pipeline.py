description = None
dataset_sample = None
title = None
agency = None
category = None
column_definitions = None
tags = None


system_message = f"""You are an assistant for a dataset search engine. Your goal
is to improve the readability of dataset descriptions for dataset search engine users."""

introduction = f"""Answer the question using the following information.

    First, consider the dataset sample:

    {dataset_sample}"""

initial_description = f"""The initial description is {description}."""

title_agency_cat = f"""Additionally the dataset title is {title}, the agency is {agency} and the category is
{category} Based on this topic and agency, please add sentence(s) describing what this
dataset can be used for."""

tag = f"""The tags are {tags}."""

column_defs = f"""Additionally, the column definitions are {column_definitions}."""

closing_instruction = f"""Question: Based on the information above and the
requirements, provide a dataset description in sentences. Use only natural,
readable sentences without special formatting."""

# read datasets.pkl
import pandas as pd
datasets = pd.read_pickle("../datasets.pkl")

new_descriptions = {}

for dataset in datasets:
  dataset_sample = dataset["data_example"]
  description = dataset['description']
  title = dataset['dataset_name']
  agency = dataset['agency']
  category = dataset['category']
  column_definitions = dataset["column_info"]
  tags = dataset['tags']
  dataset_id = dataset['dataset_id'] 

  prompt = system_message
  if dataset_sample is not None:
    prompt += introduction
  if description is not None:
    prompt += initial_description
  if title is not None:
    prompt += title_agency_cat
  if tags is not None:
    prompt += tag
  if column_definitions is not None:
    prompt += column_defs
  prompt += closing_instruction

  qwen_description = qwen_generator.generate_description(prompt)
  llama_description = llama_generator.generate_description(prompt)

  # Initialize the inner dictionary if it doesn't exist
  if dataset_id not in new_descriptions:
    new_descriptions[dataset_id] = {}

  new_descriptions[dataset_id]['qwen_description'] = qwen_description
  new_descriptions[dataset_id]['llama_description'] = llama_description


import pickle

with open('new_descriptions2.pkl', 'wb') as f:
    pickle.dump(new_descriptions, f)

