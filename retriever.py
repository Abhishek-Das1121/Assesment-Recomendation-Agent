"""
retriever.py — Two retrieval modes:
  A) Semantic: search_query from state → sentence-transformer → FAISS top-K
  B) Exact:    compare targets → fuzzy catalog lookup

Key: we now use the LLM-generated search_query directly from state,
which is far better than building it from individual fields.
"""

import json
import os
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

FAISS_PATH = os.path.join(os.path.dirname(__file__), "catalog.faiss")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "metadata.json")
MODEL_NAME = "all-MiniLM-L6-v2"


class Retriever:
    def __init__(self):
        self._model = None
        self._index = None
        self._metadata: list[dict] = []
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._load()
        self._loaded = True

    def _load(self):
        if not os.path.exists(FAISS_PATH):
            raise FileNotFoundError(f"FAISS index not found at {FAISS_PATH}. Run build_index.py first.")
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"metadata.json not found at {METADATA_PATH}. Run build_index.py first.")

        logger.info("Loading sentence-transformer model...")
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(MODEL_NAME)

        logger.info("Loading FAISS index...")
        import faiss
        self._index = faiss.read_index(FAISS_PATH)

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        logger.info(f"Retriever ready: {self._index.ntotal} vectors, {len(self._metadata)} entries")

    def build_query(self, state: dict) -> str:
        """
        Use LLM-generated search_query if available (it's better).
        Fall back to building from individual fields.
        """
        # Primary: use LLM-generated search_query
        sq = (state.get("search_query") or "").strip()
        if len(sq) >= 5:
            logger.debug(f"Using LLM search_query: {sq!r}")
            return sq

        # Fallback: build from fields
        tokens = []
        if state.get("role"):
            tokens.append(state["role"].lower())
        if state.get("seniority"):
            tokens.append(state["seniority"].lower())
        if state.get("domain"):
            tokens.append(state["domain"].lower())
        for skill in (state.get("required_skills") or []):
            tokens.append(skill.lower())

        type_map = {
            "K": "knowledge skills technical",
            "P": "personality behaviour occupational",
            "A": "cognitive ability aptitude reasoning",
            "B": "situational judgement biodata scenarios",
            "S": "simulation",
            "C": "competencies",
            "D": "development 360",
        }
        for tp in (state.get("test_type_preferences") or []):
            desc = type_map.get(tp.upper())
            if desc:
                tokens.append(desc)

        for lang in (state.get("language_constraints") or []):
            tokens.append(lang.lower())

        query = " ".join(tokens)
        logger.debug(f"Built fallback query: {query!r}")
        return query

    def semantic_search(self, query: str, top_k: int = 25) -> list[dict]:
        self._ensure_loaded()
        if not query.strip():
            return []

        vec = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = self._index.search(vec, min(top_k, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            entry = dict(self._metadata[idx])
            entry["score"] = float(score)
            results.append(entry)

        return results

    def exact_lookup(self, name: str) -> Optional[dict]:
        self._ensure_loaded()
        from catalog_engine import get_catalog_engine
        engine = get_catalog_engine()

        a = engine.get_by_name_exact(name)
        if a:
            return {
                "name": a.name, "url": a.url, "test_type": a.test_type,
                "keys": a.keys, "duration": a.duration,
                "languages": a.languages, "description": a.description, "score": 1.0,
            }

        results = engine.fuzzy_find(name, top_n=1)
        if results:
            a = results[0]
            return {
                "name": a.name, "url": a.url, "test_type": a.test_type,
                "keys": a.keys, "duration": a.duration,
                "languages": a.languages, "description": a.description, "score": 0.8,
            }
        return None

    def retrieve_for_state(self, state: dict, top_k: int = 25) -> list[dict]:
        query = self.build_query(state)
        return self.semantic_search(query, top_k=top_k)

    def retrieve_for_compare(self, targets: list[str]) -> list[Optional[dict]]:
        return [self.exact_lookup(t) for t in targets]


_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever