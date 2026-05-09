import yaml
from openai import OpenAI
import pandas as pd
import re
import json
import numpy as np
import faiss
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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


# Define function for cleaning and parsing relevant data
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

    # Strip HTML from description
    description = re.sub(
        re.compile("<.*?>"), "", str(row.get("description", ""))
    ).strip()

    # Determine string for vector store
    semantic_string = f"{categories_string} {name} {size}".strip().lower()

    return pd.Series(
        {
            "clean_name": name,
            "clean_categories": categories_string,
            "clean_size": size,
            "clean_description": description,
            "semantic_string": semantic_string,
        }
    )


# Define function for creating embeddings
def create_embeddings(texts, cache_file_path):
    """Creates embeddings in batches"""
    # Retrieve from cache
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, cache_file_path)

    if os.path.exists(cache_file):
        return np.load(cache_file)

    # Generate embeddings
    batch_size = 2000
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(input=batch, model="text-embedding-3-small")
        batch_embeddings = [x.embedding for x in response.data]
        embeddings.extend(batch_embeddings)

        # DEBUG
        print(f"Batch {i}, embeddings: {len(embeddings)}")

    embeddings = np.array(embeddings, dtype="float32")
    np.save(cache_file, embeddings)
    return embeddings


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

# Retreive or generate embeddings for both datasets
embeddings_a = create_embeddings(
    dataframe_a["semantic_string"].tolist(), "embeddings_a.npy"
)
faiss.normalize_L2(embeddings_a)

embeddings_b = create_embeddings(
    dataframe_b["semantic_string"].tolist(), "embeddings_b.npy"
)
faiss.normalize_L2(embeddings_b)

# DEBUG
print("Vectorized datasets")

# Prepare dataset b for similarity ranking and store
index = faiss.IndexFlatIP(embeddings_b.shape[1])
index.add(embeddings_b)

# Perform vector search
similarities, indices = index.search(embeddings_a, k=1)

# Build response structure
final_matches = []
llm_queries = []
records_a = dataframe_a.to_dict("records")
records_b = dataframe_b.to_dict("records")

# Iterate over records in dataset a and find closest match
for i, row_a in enumerate(records_a):
    # Validate score and append to response
    top_match_idx = indices[i][0]
    score = similarities[i][0]
    candidate_b = records_b[top_match_idx]

    if score >= 0.88:
        final_matches.append((row_a["item_id"], candidate_b["item_id"]))

    # Set aside "close enough" scores for LLM decision-making
    elif score >= 0.85:
        llm_queries.append({"score": score, "row_a": row_a, "candidate_b": candidate_b})


# DEBUG
print(f"Matches found by similarity search: {len(final_matches)}")
# DEBUG
print(f"Queries for the LLM: {len(llm_queries)}")

# Handle ambiguity if not enough matches
if len(final_matches) < 4000:
    # Sort by score
    llm_queries.sort(key=lambda x: x["score"], reverse=True)

    # Helper for running LLM query
    def evaluate_candidate(query):
        row_a = query["row_a"]
        candidate_b = query["candidate_b"]

        # Generate appropriate prompt and run query
        prompt = f"""
        You are an expert grocery merchandiser. Determine if Product A is a direct, functional equivalent to Product B.
        A match means a consumer would view these as the exact same substitute product, even if one is a private-label store brand and the other is a name brand.
        
        CRITICAL RULES:
        1. The core product must be identical.
        2. The sizes must be nearly identical. If explicit size is missing, look for weights/counts inside the product name.
        3. Ignore brand differences if they are clearly private-label substitutes.
        
        Product A:
        - Name: {row_a['clean_name']}
        - Category: {row_a['clean_categories']}
        - Size: {row_a['clean_size']}
        - Description: {row_a['clean_description'][:200]}

        Product B:
        - Name: {candidate_b['clean_name']}
        - Category: {candidate_b['clean_categories']}
        - Size: {candidate_b['clean_size']}
        - Description: {candidate_b['clean_description'][:200]}

        Respond ONLY with 'MATCH' or 'NO_MATCH'.
        """
        try: 
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            is_match = (response.choices[0].message.content.strip().upper() == 'MATCH')
        except Exception as e:
            print(f"{e}")
            is_match = False

        return (
            row_a["item_id"],
            candidate_b["item_id"],
            is_match,
        )

    # Multithread LLM requests
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all queries to the executor
        query_future = {
            executor.submit(evaluate_candidate, query): query for query in llm_queries
        }

        # Process completed requests
        for future in as_completed(query_future):
            item_a, item_b, is_match = future.result()

            if is_match:
                final_matches.append((item_a, item_b))

                # DEBUG
                print(f"LLM match: {len(final_matches)}")
            else:
                # DEBUG
                print(f"LLM non-match")

# DEBUG
print(f"Total matches found: {len(final_matches)}")

# Export CSV
matches_dataframe = pd.DataFrame(final_matches, columns=["item_id_A", "item_id_B"])
matches_dataframe.to_csv("private_label_equivalents.csv", index=False)

# DEBUG
print("Saved to private_label_equivalents.csv")
