"""
state_extractor.py — First LLM call.
Structured JSON extraction only. Temperature=0.
Uses Groq llama-3.3-70b-versatile with json_object response format.
"""

import json
import os
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

STATE_EXTRACTION_PROMPT = """You are a structured data extractor for an HR assessment system.

Extract the hiring state from the conversation. Apply cumulative overrides (newer messages override older).

Output this EXACT JSON schema, no other text:
{
  "role": "",
  "seniority": "",
  "domain": "",
  "required_skills": [],
  "excluded_skills": [],
  "test_type_preferences": [],
  "language_constraints": [],
  "compare_targets": [],
  "search_query": "",
  "has_shortlist_been_presented": false,
  "user_confirmed": false,
  "ambiguity_level": "high"
}

Field rules:
- role: job title (e.g. "software engineer", "contact centre agent", "plant operator")
- seniority: level (e.g. "senior", "graduate", "executive", "entry-level")
- domain: industry (e.g. "healthcare", "finance", "manufacturing", "sales")
- required_skills: explicitly mentioned skills/tech (e.g. ["Java", "Spring", "SQL", "AWS"])
- excluded_skills: explicitly removed items ("drop REST" → ["REST"])
- test_type_preferences: K=knowledge, P=personality, A=cognitive, B=situational judgement, S=simulation
- language_constraints: explicit language needs (e.g. ["Spanish", "Latin American Spanish"])
- compare_targets: assessment names user wants to compare (e.g. ["OPQ32r", "GSA"])
- search_query: CRITICAL — a dense retrieval query combining role + key signals. Examples:
    Contact centre agents English US → "contact centre spoken english simulation SVAR customer service"
    CXO leadership selection → "leadership executive OPQ personality benchmark director"
    Senior Java engineer → "senior java spring SQL backend knowledge technical"
    Plant operator safety → "safety dependability DSI plant operator industrial chemical"
    Graduate financial analyst → "graduate numerical reasoning finance analytical"
- has_shortlist_been_presented: true if assistant already showed recommendations
- user_confirmed: true if user explicitly said confirmed/perfect/locked/that's it
- ambiguity_level: "low" (role clear), "medium" (vague), "high" (no hiring intent)

Conversation:
{conversation}"""


def _get_client() -> Groq:
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def extract_state(messages: list[dict]) -> dict:
    conv_lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        conv_lines.append(f"{role.upper()}: {content}")
    conversation_text = "\n".join(conv_lines)

    prompt = STATE_EXTRACTION_PROMPT.replace("{conversation}", conversation_text)

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a state extraction engine. Output only valid JSON matching the schema exactly."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=600,
        )
        state = json.loads(response.choices[0].message.content.strip())
        return _validate_state(state)

    except Exception as e:
        logger.warning(f"State extraction failed: {e}. Using default.")
        return _default_state()


def _validate_state(state: dict) -> dict:
    role = str(state.get("role") or "")
    seniority = str(state.get("seniority") or "")
    required_skills = _to_str_list(state.get("required_skills"))

    # Build fallback search_query if LLM left it empty
    sq = str(state.get("search_query") or "").strip()
    if len(sq) < 5:
        sq = " ".join(filter(None, [seniority, role] + required_skills))

    return {
        "role": role,
        "seniority": seniority,
        "domain": str(state.get("domain") or ""),
        "required_skills": required_skills,
        "excluded_skills": _to_str_list(state.get("excluded_skills")),
        "test_type_preferences": _to_str_list(state.get("test_type_preferences")),
        "language_constraints": _to_str_list(state.get("language_constraints")),
        "compare_targets": _to_str_list(state.get("compare_targets")),
        "search_query": sq,
        "has_shortlist_been_presented": bool(state.get("has_shortlist_been_presented", False)),
        "user_confirmed": bool(state.get("user_confirmed", False)),
        "ambiguity_level": state.get("ambiguity_level", "medium"),
    }


def _to_str_list(val) -> list[str]:
    if not val:
        return []
    if isinstance(val, list):
        return [str(v) for v in val if v]
    return []


def _default_state() -> dict:
    return {
        "role": "",
        "seniority": "",
        "domain": "",
        "required_skills": [],
        "excluded_skills": [],
        "test_type_preferences": [],
        "language_constraints": [],
        "compare_targets": [],
        "search_query": "",
        "has_shortlist_been_presented": False,
        "user_confirmed": False,
        "ambiguity_level": "high",
    }