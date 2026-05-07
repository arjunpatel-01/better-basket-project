import pandas as pd
import faiss
import numpy as np
import json
import re
import ast
from openai import OpenAI

# Initialize OpenAI Client
client = OpenAI()

# ---------------------------------------------------------
# 1. PRE-PROCESSING & FEATURE EXTRACTION
# ---------------------------------------------------------
df_a = pd.read_csv("dataset_a.csv")
df_b = pd.read_csv("dataset_b.csv")


def clean_html(raw_html):
    """Strips HTML tags to save LLM tokens."""
    if pd.isnull(raw_html):
        return ""
    cleanr = re.compile("<.*?>")
    return re.sub(cleanr, "", str(raw_html)).strip()


def parse_tags(tags_string):
    """Safely extracts words from stringified Python lists (e.g., "['gluten_free']")"""
    if pd.isnull(tags_string):
        return ""
    try:
        # Evaluate the string as a Python list
        tags_list = ast.literal_eval(tags_string)
        # Clean up tags (remove internal underscores, etc.)
        clean_tags = [
            str(t).replace("_internal_any_", "").replace("_", " ") for t in tags_list
        ]
        return " ".join(clean_tags)
    except (ValueError, SyntaxError):
        return str(tags_string)


def build_smart_profile(row):
    """Unpacks JSON and builds a clean, brand-agnostic semantic profile."""
    # 1. Base Name
    name = str(row["name"]) if pd.notnull(row["name"]) else ""

    # 2. Unpack item_info (Categories & Ingredients)
    categories = ""
    if "item_info" in row and pd.notnull(row["item_info"]):
        try:
            info = json.loads(row["item_info"])
            cats = [info.get(f"category_{i}") for i in range(4)]
            categories = " ".join([str(c) for c in cats if c])
        except json.JSONDecodeError:
            pass

    # 3. Unpack sizing_comp (User Friendly Size)
    size = ""
    if "sizing_comp" in row and pd.notnull(row["sizing_comp"]):
        try:
            size_data = json.loads(row["sizing_comp"])
            size = size_data.get("size_user_friendly") or ""
        except json.JSONDecodeError:
            pass

    # Fallback to size_raw if JSON extraction failed
    if not size and "size_raw" in row and pd.notnull(row["size_raw"]):
        size = str(row["size_raw"])

    # 4. Extract Tags
    tags = parse_tags(row.get("tags", ""))

    # 5. Build the Brand-Agnostic Semantic String for the Vector Database
    # Format: [Categories] [Tags] [Name] [Size]
    semantic_string = f"{categories} {tags} {name} {size}".strip().lower()

    return pd.Series(
        {
            "clean_name": name,
            "clean_categories": categories,
            "clean_size": size,
            "clean_tags": tags,
            "clean_desc": clean_html(row.get("description", "")),
            "semantic_string": semantic_string,
        }
    )


print("Unpacking JSON and cleaning data...")
# Apply cleaning to both datasets
features_a = df_a.apply(build_smart_profile, axis=1)
df_a = pd.concat([df_a, features_a], axis=1)

features_b = df_b.apply(build_smart_profile, axis=1)
df_b = pd.concat([df_b, features_b], axis=1)


# ---------------------------------------------------------
# 2. EMBEDDINGS & VECTOR SEARCH (GATE 2)
# ---------------------------------------------------------
def get_embeddings_in_batches(texts, batch_size=1000):
    """Fetches embeddings in batches to avoid token limits."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        res = client.embeddings.create(input=batch, model="text-embedding-3-small")
        batch_embeddings = [x.embedding for x in res.data]
        all_embeddings.extend(batch_embeddings)
    return np.array(all_embeddings, dtype="float32")


print("Generating embeddings for Dataset B (Target)...")
b_embeddings = get_embeddings_in_batches(df_b["semantic_string"].tolist())

# Build the FAISS Index
index = faiss.IndexFlatIP(b_embeddings.shape[1])
faiss.normalize_L2(b_embeddings)
index.add(b_embeddings)

print("Generating embeddings for Dataset A (Query)...")
a_embeddings = get_embeddings_in_batches(df_a["semantic_string"].tolist())
faiss.normalize_L2(a_embeddings)

# Search FAISS for the Top 1 closest match
print("Running Vector Search...")
similarities, indices = index.search(a_embeddings, k=1)

# ---------------------------------------------------------
# 3. THE GREEDY LLM SCALPEL (GATE 3)
# ---------------------------------------------------------
final_matches = []
b_records = df_b.to_dict("records")
a_records = df_a.to_dict("records")

print("Evaluating matches...")
for i, row_a in enumerate(a_records):
    top_match_idx = indices[i][0]
    score = similarities[i][0]
    candidate_b = b_records[top_match_idx]

    # GREEDY ACCEPT: Extremely high semantic similarity (Likely exact brand or perfect clone)
    if score >= 0.94:
        final_matches.append((row_a["item_id"], candidate_b["item_id"]))
        continue

    # LLM MERCHANDISER: The "Grey Area" Equivalency Check
    elif score >= 0.78:
        prompt = f"""
        You are an expert grocery merchandiser. Determine if Product A is a direct, functional equivalent to Product B.
        A match means a consumer would view these as the exact same substitute product, even if one is a private-label store brand and the other is a name brand.
        
        CRITICAL RULES:
        1. The core product must be identical.
        2. The sizes must be nearly identical. If explicit size is missing, look for weights/counts inside the product name.
        3. Ignore brand differences (e.g., Great Value vs Wegmans vs Kraft) if they are clearly private-label substitutes.
        
        Product A:
        - Name: {row_a['clean_name']}
        - Category: {row_a['clean_categories']}
        - Tags: {row_a['clean_tags']}
        - Size: {row_a['clean_size']}
        - Description: {row_a['clean_desc'][:200]}

        Product B:
        - Name: {candidate_b['clean_name']}
        - Category: {candidate_b['clean_categories']}
        - Tags: {candidate_b['clean_tags']}
        - Size: {candidate_b['clean_size']}
        - Description: {candidate_b['clean_desc'][:200]}

        Respond ONLY with 'MATCH' or 'NO_MATCH'.
        """

        # Call gpt-5.4-nano
        response = client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # Keep it strictly deterministic
        )

        if "MATCH" in response.choices[0].message.content:
            final_matches.append((row_a["item_id"], candidate_b["item_id"]))

    # Stop once we hit the target of 4,000 matches
    if len(final_matches) >= 4000:
        print("Target of 4,000 matches reached.")
        break

# Export the final crosswalk
print(f"Total Matches Found: {len(final_matches)}")
matches_df = pd.DataFrame(final_matches, columns=["item_id_A", "item_id_B"])
matches_df.to_csv("private_label_equivalents.csv", index=False)
print("Saved to private_label_equivalents.csv")
