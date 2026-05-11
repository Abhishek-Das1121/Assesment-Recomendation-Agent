"""
response_generator.py — Second LLM call: grounded response generation only.
Uses Groq (llama-3.3-70b-versatile) — fast, high quota.
Gemini receives shortlist already determined by ranker.py — only generates text.
"""

import json
import os
import logging
import re
from groq import Groq
from dotenv import load_dotenv
from prompts import (
    SYSTEM_INSTRUCTION,
    RECOMMEND_PROMPT,
    COMPARE_PROMPT,
    CLARIFY_PROMPT,
    CONFIRM_PROMPT,
    REFUSE_PROMPT,
    CATALOG_LIMITATION_PROMPT,
)

load_dotenv()
logger = logging.getLogger(__name__)

MISSING_INFO_MAP = {
    "role": "the job role or function being hired for",
    "seniority": "the seniority or experience level",
    "domain": "the industry or functional domain",
}


def _get_client() -> Groq:
    return Groq(api_key=os.environ["YOUR_API_KEY"]) #use your API key , Groq API is prefered


def _format_conversation(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _format_shortlist_json(shortlist: list[dict]) -> str:
    simplified = [
        {
            "name": item.get("name"),
            "test_type": item.get("test_type"),
            "keys": item.get("keys"),
            "duration": item.get("duration"),
            "languages": item.get("languages", [])[:5],
            "description": (item.get("description") or "")[:200],
        }
        for item in shortlist
    ]
    return json.dumps(simplified, indent=2)


def generate_recommend_reply(
    messages: list[dict],
    state: dict,
    shortlist: list[dict],
    mode: str,
) -> str:
    if not shortlist:
        return _generate_catalog_limitation_reply(messages, state)

    prompt = RECOMMEND_PROMPT.format(
        conversation=_format_conversation(messages[-6:]),
        role=state.get("role") or "unspecified",
        seniority=state.get("seniority") or "unspecified",
        domain=state.get("domain") or "unspecified",
        skills=", ".join(state.get("required_skills") or []) or "none specified",
        languages=", ".join(state.get("language_constraints") or []) or "none",
        mode=mode,
        shortlist_json=_format_shortlist_json(shortlist),
    )
    return _call_groq(prompt)


def generate_compare_reply(messages: list[dict], compare_items: list[dict]) -> str:
    compare_context = json.dumps(
        [
            {
                "name": item.get("name"),
                "test_type": item.get("test_type"),
                "keys": item.get("keys"),
                "duration": item.get("duration"),
                "description": (item.get("description") or "")[:300],
            }
            for item in compare_items if item
        ],
        indent=2,
    )
    prompt = COMPARE_PROMPT.format(
        conversation=_format_conversation(messages[-4:]),
        compare_context=compare_context,
    )
    return _call_groq(prompt)


def generate_clarify_reply(messages: list[dict], state: dict) -> str:
    missing = []
    if not state.get("role"):
        missing.append(MISSING_INFO_MAP["role"])
    elif not state.get("seniority"):
        missing.append(MISSING_INFO_MAP["seniority"])
    elif not state.get("domain") and not state.get("required_skills"):
        missing.append("the primary skills or domain focus for this role")
    else:
        missing.append("the primary purpose: selection of new hires or development of existing staff?")

    prompt = CLARIFY_PROMPT.format(
        conversation=_format_conversation(messages[-4:]),
        role=state.get("role") or "unknown",
        seniority=state.get("seniority") or "unknown",
        skills=", ".join(state.get("required_skills") or []) or "none",
        domain=state.get("domain") or "unknown",
        ambiguity=state.get("ambiguity_level") or "high",
        missing_info="; ".join(missing),
    )
    return _call_groq(prompt)


def generate_confirm_reply(shortlist: list[dict]) -> str:
    prompt = CONFIRM_PROMPT.format(shortlist_json=_format_shortlist_json(shortlist))
    return _call_groq(prompt)


def generate_refuse_reply(user_message: str, reason: str) -> str:
    prompt = REFUSE_PROMPT.format(user_message=user_message, reason=reason)
    return _call_groq(prompt)


def _generate_catalog_limitation_reply(messages: list[dict], state: dict) -> str:
    need = " ".join([
        state.get("role") or "",
        " ".join(state.get("required_skills") or []),
    ]).strip() or "the requested skill/role"
    prompt = CATALOG_LIMITATION_PROMPT.format(need=need, alternatives_json="[]")
    return _call_groq(prompt)


def _call_groq(prompt: str) -> str:
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        return text
    except Exception as e:
        logger.error(f"Groq response generation failed: {e}")
        return "I encountered an issue generating a response. Please try again."
