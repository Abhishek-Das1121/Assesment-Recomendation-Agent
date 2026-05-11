"""
validators.py — Hallucination firewall.

Before returning ANY response, we:
1. Validate schema (Pydantic)
2. Validate every URL against catalog (hard filter)
3. Validate every assessment name against catalog
4. Strip any hallucinated entries

This is the last line of defense before JSON leaves the service.
"""

import logging
from models import ChatResponse, Recommendation
from catalog_engine import get_catalog_engine

logger = logging.getLogger(__name__)


def validate_and_sanitize_response(
    reply: str,
    recommendations: list[dict],
    end_of_conversation: bool,
) -> ChatResponse:
    """
    Build and validate a ChatResponse.
    
    - Strips recommendations with invalid URLs
    - Caps at 10 recommendations
    - Ensures schema compliance
    - Logs any stripped items (hallucination signals)
    """
    engine = get_catalog_engine()
    sanitized = []

    for item in recommendations:
        url = item.get("url", "")
        name = item.get("name", "")

        # Hard rule: URL must exist in catalog
        if not engine.is_valid_url(url):
            logger.warning(f"HALLUCINATION GUARD: Stripped invalid URL: {url!r} (name={name!r})")
            continue

        # Cross-check: name should roughly match catalog entry for this URL
        catalog_entry = engine.get_by_url(url)
        if catalog_entry and catalog_entry.name.lower() != name.lower():
            # Name mismatch: use catalog's canonical name (prevents subtle hallucinations)
            logger.warning(
                f"Name mismatch corrected: {name!r} → {catalog_entry.name!r} for {url}"
            )
            name = catalog_entry.name

        sanitized.append(
            Recommendation(
                name=name,
                url=url,
                test_type=item.get("test_type", "K"),
            )
        )

    # Cap at 10
    sanitized = sanitized[:10]

    return ChatResponse(
        reply=reply or "I can help you select SHL assessments. What role are you hiring for?",
        recommendations=sanitized,
        end_of_conversation=bool(end_of_conversation),
    )
