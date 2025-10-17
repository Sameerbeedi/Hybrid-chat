# pinecone_upload_gemini.py
"""
Upload data to Pinecone using Google Gemini embeddings (FREE)
"""
import json
import time
from tqdm import tqdm
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import config_gemini as config

# -----------------------------
# Config
# -----------------------------
DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 100  # Gemini can handle larger batches

INDEX_NAME = config.PINECONE_INDEX_NAME
VECTOR_DIM = config.PINECONE_VECTOR_DIM  # 768 for Gemini embeddings

# -----------------------------
# Initialize clients
# -----------------------------
genai.configure(api_key=config.GEMINI_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# -----------------------------
# Create index if needed
# -----------------------------
existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    print(f"Creating Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("Waiting for index to initialize...")
    time.sleep(10)
else:
    print(f"Index {INDEX_NAME} already exists.")

index = pc.Index(INDEX_NAME)

# -----------------------------
# Helper functions
# -----------------------------
def get_embeddings_batch(texts):
    """Generate embeddings using Gemini (batch)."""
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model=config.GEMINI_EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        embeddings.append(result['embedding'])
        time.sleep(0.1)  # Small delay to avoid rate limits
    return embeddings

def chunked(iterable, n):
    """Split iterable into chunks."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

# -----------------------------
# Main upload
# -----------------------------
def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    items = []
    for node in nodes:
        semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
        if not semantic_text.strip():
            continue
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", node.get("region", "")),
            "tags": node.get("tags", [])
        }
        items.append((node["id"], semantic_text, meta))

    print(f"Preparing to upsert {len(items)} items to Pinecone using Gemini embeddings...")

    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Uploading batches"):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metas = [item[2] for item in batch]

        embeddings = get_embeddings_batch(texts)

        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]

        index.upsert(vectors)
        time.sleep(0.5)

    print("✅ All items uploaded successfully!")

if __name__ == "__main__":
    main()
