import yaml
from openai import OpenAI
import pandas as pd
import re
import json
import numpy as np
import faiss

# Read the YAML file
with open("openai_creds.yaml", "r") as file:
    openai_creds = yaml.safe_load(file)

# Access the credentials
ENDPOINT = openai_creds["openai"]["endpoint"]
API_KEY = openai_creds["openai"]["api_key"]
DEPLOYMENT_NAME = openai_creds["openai"]["deployment_name"]

# DEBUG
print("Retrieved credentials")

# Initialize the client
client = OpenAI(base_url=ENDPOINT, api_key=API_KEY)

# DEBUG
print("Initialized client")


# Define functions for cleaning and parsing relevant data
def strip_html(html):
    """Strips HTML tags (reduce tokens)"""
    if pd.isnull(html):
        return ""
    return re.sub(re.compile("<.*?>"), "", str(html)).strip()


def build_relevant_data(row):
    """Clean dataset and build relevant data for embeddings"""
    # Extract name
    name = str(row["name"]) if pd.notnull(row["name"]) else ""

    # Unpack and extract categories
    categories_string = ""
    if "item_info" in row and pd.notnull(row["item_info"]):
        try:
            info = json.loads(row["item_info"])
            categories = [info.get(f"category_{i}") for i in range(4)]
            categories_string = " ".join([str(c) for c in categories if c])
        except json.JSONDecodeError:
            pass

    # Unpack and extract size
    size = ""
    if "sizing_comp" in row and pd.notnull(row["sizing_comp"]):
        try:
            size_data = json.loads(row["sizing_comp"])

            if isinstance(size_data, dict):
                size = size_data.get("size_user_friendly") or ""
            else:
                size = ""
        except json.JSONDecodeError:
            pass

    # Determine string for vector store
    semantic_string = f"{categories_string} {name} {size}".strip().lower()

    return pd.Series(
        {
            "clean_name": name,
            "clean_categories": categories_string,
            "clean_size": size,
            "clean_desc": strip_html(row.get("description", "")),
            "semantic_string": semantic_string,
        }
    )


# Define function for creating embeddings
def create_embeddings(texts, batch_size=2000):
    """Creates embeddings in batches"""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            input=batch, model="text-embedding-3-small"
        )
        batch_embeddings = [x.embedding for x in response.data]
        embeddings.extend(batch_embeddings)

        # DEBUG
        print(f"Batch {i}, embeddings: {len(embeddings)}")

    return np.array(embeddings, dtype="float32")


# Read data
dtype_mapping = {"item_id": str, "tags": str, "raw_data_id": str}
dataframe_a = pd.read_csv("grocery_store_a_items_final.csv", dtype=dtype_mapping)
dataframe_b = pd.read_csv("grocery_store_b_items_final.csv", dtype=dtype_mapping)

# DEBUG
print("Read datasets")

# Clean both datasets and overwrite with relevant data
cleaned_a = dataframe_a.apply(build_relevant_data, axis=1)
dataframe_a = pd.concat([dataframe_a, cleaned_a], axis=1)

cleaned_b = dataframe_b.apply(build_relevant_data, axis=1)
dataframe_b = pd.concat([dataframe_b, cleaned_b], axis=1)

# DEBUG
print("Cleaned datasets")

# Vectorize both datasets
embeddings_a = create_embeddings(dataframe_a["semantic_string"].tolist())
faiss.normalize_L2(embeddings_a)

embeddings_b = create_embeddings(dataframe_b["semantic_string"].tolist())
faiss.normalize_L2(embeddings_b)

# DEBUG
print("Vectorized datasets")

# Prepare dataset b for similarity ranking and store
index = faiss.IndexFlatIP(embeddings_b.shape[1])
index.add(embeddings_b)  # this may require a second argument, double check

# Perform vector search
similarities, indices = index.search(embeddings_a, k=1)

# Build response structure
final_matches = []
records_a = dataframe_a.to_dict("records")
records_b = dataframe_b.to_dict("records")

# Iterate over records in dataset a and find closest match
for i, row_a in enumerate(records_a):
    # Validate score and append to response
    top_match_idx = indices[i][0]
    score = similarities[i][0]
    candidate_b = records_b[top_match_idx]

    if score >= 0.90:
        final_matches.append((row_a["item_id"], candidate_b["item_id"]))

# DEBUG
print(f"Total Matches Found: {len(final_matches)}")

# Export CSV
matches_dataframe = pd.DataFrame(final_matches, columns=["item_id_A", "item_id_B"])
matches_dataframe.to_csv("private_label_equivalents.csv", index=False)

# DEBUG
print("Saved to private_label_equivalents.csv")
