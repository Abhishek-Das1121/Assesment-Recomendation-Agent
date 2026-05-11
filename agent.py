"""
agent.py — Main pipeline orchestrator.

Flow per request:
  messages[]
    → message preprocessing
    → Gemini state extraction
    → Python mode detection
    → retrieval strategy selection
    → FAISS retrieval / exact lookup
    → ranking + filtering
    → grounded Gemini response generation
    → schema + URL validation
    → ChatResponse

This module is the only place that calls all other modules.
main.py calls only this.
"""

import json
import logging
import re
from models import ChatRequest, ChatResponse
from state_extractor import extract_state
from mode_detector import detect_mode
from retriever import get_retriever
from ranker import rank_and_filter
from response_generator import (
    generate_recommend_reply,
    generate_compare_reply,
    generate_clarify_reply,
    generate_confirm_reply,
    generate_refuse_reply,
)
from validators import validate_and_sanitize_response

logger = logging.getLogger(__name__)


def process_chat(request: ChatRequest) -> ChatResponse:
    """
    Full stateless pipeline. Called once per /chat request.
    Reconstructs all state from messages[].
    """
    messages = [m.model_dump() for m in request.messages]

    # --- Step 1: Preprocess messages ---
    clean_messages = preprocess_messages(messages)

    # --- Step 2: Count turns (user+assistant pairs) ---
    turn_count = sum(1 for m in messages if m["role"] == "user")

    # --- Step 3: Extract structured state via Gemini ---
    state = extract_state(clean_messages)
    logger.info(f"Extracted state: role={state.get('role')!r}, "
                f"seniority={state.get('seniority')!r}, "
                f"mode_hint=ambiguity={state.get('ambiguity_level')!r}, "
                f"turn={turn_count}")

    # --- Step 4: Detect mode (pure Python) ---
    mode = detect_mode(state, messages, turn_count)
    logger.info(f"Detected mode: {mode}")

    # --- Step 5: Reconstruct current shortlist from message history ---
    # The API is stateless, so we parse prior assistant recommendations
    # from the message history. The evaluator sends full history each turn.
    current_shortlist = extract_shortlist_from_messages(messages)

    # --- Step 6: Branch by mode ---
    retriever = get_retriever()

    if mode == "refuse":
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        reply = generate_refuse_reply(
            last_user,
            reason="This question is outside the scope of SHL assessment recommendation.",
        )
        return validate_and_sanitize_response(reply, [], False)

    if mode == "compare":
        targets = state.get("compare_targets") or []
        compare_items = retriever.retrieve_for_compare(targets)
        found = [item for item in compare_items if item]
        reply = generate_compare_reply(clean_messages, found)
        # Compare does not change the shortlist
        return validate_and_sanitize_response(reply, current_shortlist, False)

    if mode == "confirm":
        reply = generate_confirm_reply(current_shortlist)
        return validate_and_sanitize_response(reply, current_shortlist, True)

    if mode == "clarify":
        reply = generate_clarify_reply(clean_messages, state)
        return validate_and_sanitize_response(reply, [], False)

    if mode in ("recommend", "refine"):
        # Retrieve candidates
        candidates = retriever.retrieve_for_state(state, top_k=25)

        # Rank + filter + diversify
        shortlist = rank_and_filter(
            candidates=candidates,
            state=state,
            current_shortlist=current_shortlist,
            mode=mode,
        )

        # Generate grounded reply
        reply = generate_recommend_reply(
            messages=clean_messages,
            state=state,
            shortlist=shortlist,
            mode=mode,
        )

        return validate_and_sanitize_response(reply, shortlist, False)

    # Fallback (should never reach)
    logger.error(f"Unhandled mode: {mode}")
    return validate_and_sanitize_response(
        "I can help you select SHL assessments. What role are you hiring for?",
        [],
        False,
    )


def preprocess_messages(messages: list[dict]) -> list[dict]:
    """
    Clean message list before sending to Gemini state extractor.
    
    - Remove pure filler turns ("ok", "thanks", etc.)
    - Remove repeated recommendation table dumps from assistant messages
    - Keep the last N turns to manage token budget
    
    We keep the user messages intact since they carry the signal.
    We truncate assistant messages to avoid polluting the extraction prompt.
    """
    MAX_TURNS_FOR_EXTRACTION = 12  # max messages sent to state extractor

    cleaned = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "").strip()

        if role == "user":
            # Keep all user messages (even short ones like "Yes" or "US")
            cleaned.append({"role": role, "content": content})
        else:
            # For assistant messages: strip markdown table noise, keep first 300 chars
            # The state extractor doesn't need the full table
            condensed = _condense_assistant_message(content)
            if condensed:
                cleaned.append({"role": role, "content": condensed})

    # Keep last N messages for token efficiency
    return cleaned[-MAX_TURNS_FOR_EXTRACTION:]


def _condense_assistant_message(content: str) -> str:
    """
    Strip markdown recommendation tables from assistant messages.
    Keep only the prose explanation (first paragraph).
    """
    lines = content.split("\n")
    prose_lines = []
    for line in lines:
        # Skip table rows and headers
        if line.strip().startswith("|") or line.strip().startswith("---"):
            continue
        # Skip URLs inside markdown
        if re.match(r"^\s*https?://", line.strip()):
            continue
        prose_lines.append(line)

    condensed = "\n".join(prose_lines).strip()
    # Truncate long assistant messages
    return condensed[:400] if condensed else ""


def extract_shortlist_from_messages(messages: list[dict]) -> list[dict]:
    """
    Reconstruct the last shortlist from assistant messages.
    
    The evaluator sends full conversation history. We parse the last
    assistant message that contained recommendations.
    
    We look for markdown table rows with URLs in the SHL catalog format.
    This is the only reliable source of the "current shortlist" in a
    stateless architecture.
    """
    from catalog_engine import get_catalog_engine
    engine = get_catalog_engine()

    # Walk messages in reverse to find the last recommendation table
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue

        content = msg.get("content", "")
        urls_found = re.findall(
            r"https://www\.shl\.com/products/product-catalog/view/[^\s\)>\|]+",
            content,
        )
        if not urls_found:
            continue

        shortlist = []
        seen = set()
        for url in urls_found:
            url = url.rstrip("/)>")
            if url in seen:
                continue
            seen.add(url)
            entry = engine.get_by_url(url)
            if entry:
                shortlist.append({
                    "name": entry.name,
                    "url": entry.url,
                    "test_type": entry.test_type,
                    "keys": entry.keys,
                    "duration": entry.duration,
                    "languages": entry.languages,
                    "description": entry.description,
                    "score": 1.0,
                })
        if shortlist:
            return shortlist

    return []
