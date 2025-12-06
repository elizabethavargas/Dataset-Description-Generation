# imports
import requests
import pandas as pd
from io import BytesIO
import pickle
import json
from difflib import SequenceMatcher

def list_nyc_open_data_datasets():
  """
    Fetches all datasets from the NYC Open Data Socrata API.

    Returns:
        list[dict]: List of {"id": <id>, "name": <name>} dictionaries.
                    Returns [] if any error occurs.
    """

  # Base URL for the NYC Open Data API
  base_url = "https://data.cityofnewyork.us/api/views.json"

  try:
      response = requests.get(base_url)
      response.raise_for_status()  # Raise an exception for bad status codes
      datasets_data = response.json()

      # Extract id and name for each dataset
      datasets_list = []
      for dataset in datasets_data:
          if 'id' in dataset and 'name' in dataset:
              datasets_list.append({'id': dataset['id'], 'name': dataset['name']})

      # Print confirmation message
      print(f"Successfully listed {len(datasets_list)} datasets.")

  except requests.exceptions.RequestException as e:
      print(f"Error fetching data: {e}")
  except ValueError:
      print("Error decoding JSON response. The response might not be in JSON format.")

  return datasets_list


def find_closest_match(candidates, references):
    """
    Returns the candidate string that is most similar to any of the reference strings.

    Args:
        candidates (list[str]): List of strings to choose from.
        references (list[str]): List of reference strings to compare against.

    Returns:
        str | None: The candidate with highest similarity to any reference,
                    or None if candidates is empty.
    """
    if not candidates:
        return None

    best_match = None
    highest_similarity = -1

    for candidate in candidates:
        for reference in references:
            similarity = SequenceMatcher(
                None,
                candidate.lower(),
                reference.lower()
            ).ratio()

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = candidate

    return best_match


def format_example_rows(df, max_chars=150):
    """
    Formats the first n rows of a DataFrame as clean JSON-like examples for LLM prompts.
    - Truncates long strings/numbers.
    - Removes NaN values.
    """
    example = {}
    for col, val in df.iloc[0].items():
        # Simplify nested dicts or long text
        if isinstance(val, (dict, list)):
            val_str = json.dumps(val)
        else:
            val_str = str(val)
        if len(val_str) > max_chars:
            val_str = val_str[:max_chars] + "..."
        if val_str not in ("nan", "None", ""):
            example[col] = val_str
    return example


def fetch_dataset_info(dataset_id, app_token = "L76aBvmvvFwme9Q46GQJ3qtf8"):
    """
    Fetches data example, metadata, tags, column descriptions, and data dictionary
    for a single NYC Open Data dataset.

    Args:
        dataset_id (dict): A string of the dataset id found in the URL of the dataset.

    Returns:
        dict or None: A dataset_info object, or None if something failed.
    """


    print(f"\n--- Querying dataset (ID: {dataset_id}) ---")


    # URLs
    dataset_url = f"https://data.cityofnewyork.us/resource/{dataset_id}.json"
    metadata_url = f"https://data.cityofnewyork.us/api/views/{dataset_id}.json"

    # FETCH SAMPLE DATA
    try:
        data_response = requests.get(
            dataset_url,
            headers={"X-App-Token": app_token},
            params={'$limit': 2}
        )
        data_response.raise_for_status()
        data = data_response.json()

        if not data:
            print(f"No data returned for dataset: {dataset_id}")
            return None

        df = pd.DataFrame(data)
        if df.empty:
            print(f"No rows returned for dataset: {dataset_id}")
            return None

        data_example = format_example_rows(df)

    except Exception as e:
        print(f"Error retrieving sample data for {dataset_id}: {e}")
        return None

    # FETCH METADATA
    try:
        metadata_response = requests.get(metadata_url)
        metadata_response.raise_for_status()
        metadata = metadata_response.json()

        if not metadata:
            print("No metadata returned.")
            return None

        # Basic metadata
        dataset_name = metadata.get("name", "N/A")
        category = metadata.get("category", "N/A")
        description_raw = metadata.get("description", "")
        description = "\n".join([line for line in description_raw.splitlines() if line.strip()])
        agency = metadata.get("metadata", {}).get("custom_fields", {})\
                        .get("Dataset Information", {}).get("Agency", "N/A")
        tags = metadata.get("tags", [])
        column_info = {}

        # COLUMN DESCRIPTIONS FROM METADATA
        for col in metadata.get("columns", []):
            name = col.get("name", "")
            desc = col.get("description", "")
            if desc:
                column_info[name] = desc


        # TRY DATA DICTIONARY FILE IF METADATA HAS NO COLUMN INFO
        attachments = metadata.get("metadata", {}).get("attachments", [])
        if attachments and not column_info:
            print("COLUMN INFO FROM DATA DICTIONARY FILE")

            # Find closest file match
            filenames = [a["filename"] for a in attachments]
            match = find_closest_match(
                filenames,
                ["data dictionary", "column descriptions", "column definitions"]
            )

            if match:
                attach = next(a for a in attachments if a["filename"] == match)
                file_id = attach.get("assetId")
                file_url = (
                    f"https://data.cityofnewyork.us/api/views/{dataset_id}/files/"
                    f"{file_id}?download=true&filename={match}"
                )
                print(f"Downloading attached dictionary: {match}")

                try:
                    file_response = requests.get(file_url)
                    file_response.raise_for_status()

                    excel_file = BytesIO(file_response.content)
                    xls = pd.ExcelFile(excel_file)

                    # pick best sheet
                    sheet = find_closest_match(
                        xls.sheet_names,
                        ["data dictionary", "column descriptions", "column definitions"]
                    )

                    df_preview = pd.read_excel(xls, sheet_name=sheet, header=None)

                    # Detect header row
                    def detect_header_row(df_full):
                        for i in range(min(10, len(df_full))):
                            row = df_full.iloc[i].astype(str).str.lower()
                            keywords = ["column", "name", "description", "field"]
                            matches = sum(any(k in cell for k in keywords) for cell in row)
                            if matches >= 2:
                                return i
                        return 0

                    header = detect_header_row(df_preview)
                    df_dict = pd.read_excel(xls, sheet_name=sheet, header=header)

                    # Match columns
                    name_col = find_closest_match(
                        df_dict.columns,
                        ["column", "column name", "field", "field name"]
                    )
                    desc_col = find_closest_match(
                        df_dict.columns,
                        ["description", "definition", "column description"]
                    )

                    if name_col and desc_col:
                        for _, row in df_dict.iterrows():
                            name_val = str(row[name_col]).strip()
                            desc_val = str(row[desc_col]).strip()
                            if desc_val:
                                column_info[name_val] = desc_val

                except Exception as e:
                    print(f"Error processing dictionary file: {e}")

    except Exception as e:
        print(f"Error fetching metadata for {dataset_name}: {e}")
        return None

    # RETURN FINAL OBJECT
    dataset_info = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "data_example": data_example,
        "category": category,
        "description": description,
        "agency": agency,
        "tags": tags,
        "column_info": column_info,
    }

    print(f"Finished dataset: {dataset_name}")

    return dataset_info

def fetch_all_datasets(output_path="datasets.pkl", app_token="L76aBvmvvFwme9Q46GQJ3qtf8"):
    """
    Fetches all NYC Open Data datasets and saves detailed info to a pickle file.

    Parameters
    ----------
    output_path : str
        Where to save the pickle file (default: 'datasets.pkl')

    Returns
    -------
    list
        A list of dataset info dicts (also saved to disk)
    """
    datasets_list = list_nyc_open_data_datasets()
    datasets = []

    for dataset in datasets_list:
        try:
            info = fetch_dataset_info(dataset['id'], app_token)
            if info:
                datasets.append(info)
        except Exception as e:
            print(f"⚠️ Skipping dataset {dataset.get('id', '?')}: {e}")
            continue

    # Save result
    with open(output_path, "wb") as f:
        pickle.dump(datasets, f)

    return datasets
