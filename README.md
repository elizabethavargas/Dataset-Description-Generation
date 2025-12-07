# Dataset-Description-Generation
This project uses large language models (LLMs) to generate enhanced descriptions for datasets hosted on NYC Open Data. These improved descriptions aim to help users more easily find datasets and assess their relevance.

---

## Installation

Clone the repository and install dependencies via pip:

```bash
git clone https://github.com/elizabethavargas/Dataset-Description-Generation.git
pip install -r Dataset-Description-Generation/requirements.txt
```

---

## Getting Started

The simplest way to use AutoDDG is to create an instance and generate a dataset description:

```python
import sys
sys.path.append('/content/Dataset-Description-Generation')
from src.generator.generator import HFGenerator
from src.utils.nyc_utils import fetch_dataset_info

# fetch dataset info using an id from any NYC Open Data dataset
test_dataset = fetch_dataset_info('8wbx-tsch')
test_dataset

# generate description
llama_generator = HFGenerator("unsloth/Meta-Llama-3.1-8B-Instruct")
test_description = llama_generator.generate_description(test_dataset, user_focused=True)

print(description)
# >>> This dataset, For Hire Vehicles (FHV) - Active, is provided by the Taxi and Limousine Commission (TLC) and contains information about active for-hire vehicles in New York City. The dataset includes details such as vehicle license numbers, owner names, license types, expiration dates, and base information. It is updated daily between 4-7pm and can be used to track the status of for-hire vehicles, identify active drivers, and monitor the taxi industry in New York City. This dataset can be used by researchers, policymakers, and industry professionals to analyze trends, identify patterns, and make informed decisions about the for-hire vehicle industry.
```

### Quick Jupyter Notebook Start

For a much better introduction, we **highly recommend** starting with the quick_start notebook.
