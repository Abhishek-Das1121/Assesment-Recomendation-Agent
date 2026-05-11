"""
mode_detector.py — Pure Python deterministic mode detection.
NO LLM here. Scoring logic only.
"""

import re
import logging
from typing import Literal

logger = logging.getLogger(__name__)

Mode = Literal["compare", "refuse", "confirm", "refine", "recommend", "clarify"]

# Off-topic signals that should trigger refusal
REFUSE_PATTERNS = [
    r"\blegal\b", r"\blaw\b", r"\bcompliance\b", r"\bregulat\b",
    r"\bhipaa requirement\b", r"\bgdpr\b", r"\blawsuit\b", r"\bliabilit\b",
    r"\bdisgram\b", r"\bignore (previous|all|your) instructions\b",
    r"\bforget (your|all) instructions\b", r"\bact as\b",
    r"\bpretend\b", r"\byou are now\b",
    r"\bgeneral hiring advice\b", r"\bcompensation\b", r"\bsalar\b",
    r"\binterview (tips|coaching)\b",
    r"\bprompt injection\b",
]

# Refined confirmation patterns to be less "greedy"
CONFIRM_PATTERNS = [
    r"\bconfirm\b", r"\bfinalized?\b", r"\bsettled?\b", 
    r"\bfinal (list|battery|shortlist)\b",
    r"\blocking (it|them) in\b", r"\bsound(s)? good\b",
    r"\bthat works?\b", r"\byes\b", r"\bgo ahead\b"
]

# Signals that the user wants to refine the shortlist
REFINE_PATTERNS = [
    r"\badd\b", r"\binclude\b", r"\bremove\b", r"\bdrop\b", r"\bexclude\b",
    r"\bswap\b", r"\breplace\b", r"\binstead\b", r"\balso add\b",
    r"\bwithout\b", r"\bnot (the|this)\b", r"\bkeep\b.*\bbut\b",
    r"\bskip (the|this)\b", r"\bno (need|personality|cognitive)\b",
    r"\bactually\b", r"\bwait\b", r"\bchange\b", r"\bupdate\b",
]

def detect_mode(
    state: dict,
    messages: list[dict],
    turn_count: int,
) -> Mode:
    last_user_msg = _get_last_user_message(messages).lower()

    # 1. COMPARE
    if state.get("compare_targets"):
        return "compare"

    # 2. REFUSE
    if _matches_any(last_user_msg, REFUSE_PATTERNS):
        if not state.get("role") and not state.get("has_shortlist_been_presented"):
            return "refuse"
        if re.search(r"\blegal\b|\bcompl[yi]\b|\brequired (by law|under)\b", last_user_msg):
            return "refuse"

    # 3. REFINE - Higher priority than confirm to catch "Perfect, but add X"
    if (
        state.get("has_shortlist_been_presented")
        and _matches_any(last_user_msg, REFINE_PATTERNS)
    ):
        return "refine"

    # 4. CONFIRM
    if (
        state.get("has_shortlist_been_presented")
        and _matches_any(last_user_msg, CONFIRM_PATTERNS)
    ):
        return "confirm"

    # 5. RECOMMEND - Lowered threshold (2) and Turn Limit (2)
    readiness = compute_readiness_score(state)
    if readiness >= 2 or turn_count >= 2:
        return "recommend"

    # 6. CLARIFY
    return "clarify"

def compute_readiness_score(state: dict) -> int:
    """
    Lowered threshold logic: Role is now a +3. 
    A single Role signal now triggers a recommendation.
    """
    score = 0

    if state.get("role"):
        score += 3  # Boosted from 2 to 3

    if state.get("seniority"):
        score += 1

    if state.get("required_skills"):
        score += 2

    if state.get("domain"):
        score += 1

    if state.get("test_type_preferences"):
        score += 1

    # Penalize ambiguity
    if state.get("ambiguity_level") == "high":
        score -= 2
    elif state.get("ambiguity_level") == "medium":
        score -= 1

    return score

def _get_last_user_message(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""

def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)