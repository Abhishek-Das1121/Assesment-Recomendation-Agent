"""
catalog_engine.py — Loads catalog.json and builds all lookup structures.

Responsibilities:
- Load and validate catalog entries
- Build name→entry and url→entry maps for exact lookup
- Build fuzzy-name index for compare mode
- Expose clean typed access to the rest of the pipeline

This module is imported once at startup. All lookups are O(1) or O(n) 
over the small catalog (~200 entries). No DB needed.
"""

import json
import re
import os
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")


@dataclass
class Assessment:
    name: str
    url: str
    test_type: str          # e.g. "K" or "K,S"
    keys: list[str]         # e.g. ["Knowledge & Skills", "Simulations"]
    duration: str           # e.g. "9 minutes" or ""
    languages: list[str]
    description: str
    # computed at load time
    name_normalized: str = field(default="", repr=False)
    search_text: str = field(default="", repr=False)  # for embedding

    def to_recommendation(self) -> dict:
        return {
            "name": self.name,
            "url": self.url,
            "test_type": self.test_type,
        }

    def primary_type_codes(self) -> list[str]:
        return [c.strip() for c in self.test_type.split(",") if c.strip()]

    def has_language(self, lang: str) -> bool:
        lang_lower = lang.lower()
        return any(lang_lower in l.lower() for l in self.languages)


class CatalogEngine:
    def __init__(self, catalog_path: str = CATALOG_PATH):
        self.assessments: list[Assessment] = []
        self._by_url: dict[str, Assessment] = {}
        self._by_name_normalized: dict[str, Assessment] = {}
        self._load(catalog_path)

    def _normalize_name(self, name: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace."""
        n = name.lower()
        n = re.sub(r"[^a-z0-9 ]", " ", n)
        n = re.sub(r"\s+", " ", n).strip()
        return n

    def _build_search_text(self, a: Assessment) -> str:
        """Concatenate all searchable fields into one string for embedding."""
        parts = [
            a.name,
            a.description,
            " ".join(a.keys),
            " ".join(a.languages[:10]),
            a.duration,
        ]
        return " ".join(p for p in parts if p).strip()

    def _load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"catalog.json not found at {path}. "
                "Run scrape_catalog.py first."
            )
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        loaded = 0
        skipped = 0
        for entry in raw:
            name = (entry.get("name") or "").strip()
            url = (entry.get("url") or "").strip()
            if not name or not url:
                skipped += 1
                continue
            if not url.startswith("https://www.shl.com"):
                skipped += 1
                continue

            a = Assessment(
                name=name,
                url=url,
                test_type=entry.get("test_type") or "K",
                keys=entry.get("keys") or [],
                duration=entry.get("duration") or "",
                languages=entry.get("languages") or [],
                description=entry.get("description") or "",
            )
            a.name_normalized = self._normalize_name(name)
            a.search_text = self._build_search_text(a)

            self.assessments.append(a)
            self._by_url[url] = a
            self._by_name_normalized[a.name_normalized] = a
            loaded += 1

        logger.info(f"CatalogEngine: loaded {loaded} assessments, skipped {skipped}")
        if loaded == 0:
            raise ValueError("Catalog is empty after loading. Check catalog.json.")

    def get_by_url(self, url: str) -> Optional[Assessment]:
        return self._by_url.get(url)

    def get_by_name_exact(self, name: str) -> Optional[Assessment]:
        return self._by_name_normalized.get(self._normalize_name(name))

    def fuzzy_find(self, query: str, top_n: int = 3) -> list[Assessment]:
        """
        Find assessments whose normalized name contains all tokens
        from query, or partial token overlap. Used in compare mode.
        """
        q_tokens = set(self._normalize_name(query).split())
        if not q_tokens:
            return []

        scored = []
        for a in self.assessments:
            name_tokens = set(a.name_normalized.split())
            overlap = len(q_tokens & name_tokens)
            if overlap > 0:
                score = overlap / max(len(q_tokens), 1)
                scored.append((score, a))

        scored.sort(key=lambda x: -x[0])
        return [a for _, a in scored[:top_n]]

    def is_valid_url(self, url: str) -> bool:
        return url in self._by_url

    def all_assessments(self) -> list[Assessment]:
        return self.assessments

    def filter_by_type(self, type_codes: list[str]) -> list[Assessment]:
        """Return assessments matching any of the given type codes."""
        if not type_codes:
            return self.assessments
        result = []
        for a in self.assessments:
            for code in a.primary_type_codes():
                if code in type_codes:
                    result.append(a)
                    break
        return result


# Module-level singleton — imported by other modules
_engine: Optional[CatalogEngine] = None


def get_catalog_engine() -> CatalogEngine:
    global _engine
    if _engine is None:
        _engine = CatalogEngine()
    return _engine
