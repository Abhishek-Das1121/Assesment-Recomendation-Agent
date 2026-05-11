"""
build_index.py — One-time script to embed catalog entries and save FAISS index.

Run AFTER scrape_catalog.py:
    python build_index.py

Outputs:
    catalog.faiss   — FAISS flat L2 index
    metadata.json   — ordered list of assessment metadata (parallel to index rows)

Model: all-MiniLM-L6-v2 (22MB, CPU-fast, good semantic quality for short texts)
"""

import json
import os
import pickle
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CATALOG_PATH = "catalog.json"
FAISS_PATH = "catalog.faiss"
METADATA_PATH = "metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"


def build_search_text(entry: dict) -> str:
    """
    Combine name + description + keys into one string for embedding.
    Do NOT embed raw conversation. Embed catalog-derived text only.
    """
    parts = [
        entry.get("name", ""),
        entry.get("description", ""),
        " ".join(entry.get("keys", [])),
    ]
    return " ".join(p for p in parts if p).strip()


def main():
    # --- Load catalog ---
    if not os.path.exists(CATALOG_PATH):
        raise FileNotFoundError("catalog.json not found. Run scrape_catalog.py first.")

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    logger.info(f"Loaded {len(catalog)} entries from catalog.json")

    # --- Build text corpus ---
    texts = [build_search_text(e) for e in catalog]
    metadata = [
        {
            "name": e.get("name", ""),
            "url": e.get("url", ""),
            "test_type": e.get("test_type", "K"),
            "keys": e.get("keys", []),
            "duration": e.get("duration", ""),
            "languages": e.get("languages", []),
            "description": e.get("description", ""),
        }
        for e in catalog
    ]

    # --- Embed ---
    logger.info(f"Loading sentence-transformer model: {MODEL_NAME}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    logger.info("Embedding catalog entries (CPU)...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity via dot product
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)
    logger.info(f"Embedding shape: {embeddings.shape}")

    # --- Build FAISS index ---
    import faiss
    dim = embeddings.shape[1]
    # IndexFlatIP: exact inner product (= cosine on normalized vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    # --- Save ---
    faiss.write_index(index, FAISS_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved index → {FAISS_PATH}")
    logger.info(f"Saved metadata → {METADATA_PATH}")
    logger.info("Index build complete.")


if __name__ == "__main__":
    main()
