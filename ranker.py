"""
ranker.py — Deterministic ranking, filtering, and shortlist management.

Pipeline:
  1. Hard filters (excluded skills)
  2. Hybrid scoring (semantic + skill overlap + type alignment + anchor boost)
  3. Role-specific anchor injection
  4. Diversification (max per type)
  5. Refinement preservation
  6. Cap at 10 items
"""

import logging

logger = logging.getLogger(__name__)

MAX_PER_TYPE = 4

# Universal score boosts
ANCHOR_BOOSTS: dict[str, float] = {
    "occupational personality questionnaire opq32r": 2.5,
    "shl verify interactive g+": 1.8,
    "shl verify interactive g": 1.8,
}

SENIOR_SENIORITY_KEYWORDS = {
    "senior", "lead", "principal", "staff", "architect",
    "manager", "director", "executive", "cxo", "vp", "head",
    "graduate", "grad",
}

# Role keyword → catalog name anchors to always inject
ROLE_ANCHORS: list[tuple[list[str], list[str]]] = [
    (
        ["contact centre", "contact center", "customer service"],
        ["svar spoken english (us) (new)", "contact center call simulation (new)"],
    ),
    (
        ["plant operator", "chemical", "safety", "industrial operator"],
        ["workplace health and safety (new)", "dependability and safety instrument (dsi)"],
    ),
    (
        ["leadership", "cxo", "director", "executive"],
        ["opq leadership report", "opq universal competency report 2.0"],
    ),
    (
        ["financial analyst", "finance", "financial"],
        ["financial accounting (new)", "basic statistics (new)"],
    ),
    (
        ["graduate", "management trainee"],
        ["graduate scenarios"],
    ),
]


def rank_and_filter(
    candidates: list[dict],
    state: dict,
    current_shortlist: list[dict],
    mode: str,
) -> list[dict]:
    if mode == "refine" and current_shortlist:
        return _refine_shortlist(candidates, state, current_shortlist)
    else:
        return _fresh_recommendation(candidates, state)


def _fresh_recommendation(candidates: list[dict], state: dict) -> list[dict]:
    excluded = {s.lower() for s in (state.get("excluded_skills") or [])}
    required_skills = [s.lower() for s in (state.get("required_skills") or [])]
    type_prefs = [t.upper() for t in (state.get("test_type_preferences") or [])]
    lang_constraints = [l.lower() for l in (state.get("language_constraints") or [])]
    seniority = (state.get("seniority") or "").lower()
    role = (state.get("role") or "").lower()
    search_query = (state.get("search_query") or "").lower()
    is_senior = any(kw in seniority for kw in SENIOR_SENIORITY_KEYWORDS)

    filtered = [c for c in candidates if not _matches_excluded(c, excluded)]

    scored = []
    for c in filtered:
        score = _compute_score(c, state, required_skills, type_prefs, lang_constraints, is_senior)
        scored.append((score, c))

    scored.sort(key=lambda x: -x[0])

    shortlist = _diversify(scored, type_prefs)
    shortlist = _inject_anchors(shortlist, candidates, excluded, is_senior, role, search_query)

    return shortlist[:10]


def _refine_shortlist(
    candidates: list[dict],
    state: dict,
    current_shortlist: list[dict],
) -> list[dict]:
    excluded = {s.lower() for s in (state.get("excluded_skills") or [])}
    required_skills = [s.lower() for s in (state.get("required_skills") or [])]

    preserved = [item for item in current_shortlist if not _matches_excluded(item, excluded)]
    preserved_names = {item["name"].lower() for item in preserved}

    new_candidates = []
    for c in candidates:
        if c["name"].lower() not in preserved_names:
            if not _matches_excluded(c, excluded):
                skill_match = _skill_overlap_score(c, required_skills)
                if skill_match > 0:
                    new_candidates.append((skill_match + c.get("score", 0), c))

    new_candidates.sort(key=lambda x: -x[0])

    merged = list(preserved)
    existing_names = {item["name"].lower() for item in merged}

    for _, c in new_candidates:
        if c["name"].lower() not in existing_names and len(merged) < 10:
            merged.append(c)
            existing_names.add(c["name"].lower())

    return merged[:10]


def _compute_score(
    candidate: dict,
    state: dict,
    required_skills: list[str],
    type_prefs: list[str],
    lang_constraints: list[str],
    is_senior: bool,
) -> float:
    name_lower = (candidate.get("name") or "").lower()

    score = candidate.get("score", 0.0) * 3.0

    # Anchor boost
    anchor_boost = ANCHOR_BOOSTS.get(name_lower, 0.0)
    if anchor_boost > 0:
        if "verify" in name_lower and not is_senior:
            anchor_boost = 0.0
        score += anchor_boost

    score += _skill_overlap_score(candidate, required_skills) * 2.0

    if type_prefs:
        candidate_types = [t.strip() for t in candidate.get("test_type", "").split(",")]
        if any(t in type_prefs for t in candidate_types):
            score += 2.0

    if lang_constraints:
        candidate_langs = " ".join(candidate.get("languages", [])).lower()
        if any(l in candidate_langs for l in lang_constraints):
            score += 1.5
        else:
            score -= 1.0

    description = (candidate.get("description") or "").lower()
    seniority = (state.get("seniority") or "").lower()
    role = (state.get("role") or "").lower()

    if seniority and seniority in (description + " " + name_lower):
        score += 0.5
    if role:
        if any(t in name_lower for t in role.split()):
            score += 0.5

    return score


def _inject_anchors(
    shortlist: list[dict],
    all_candidates: list[dict],
    excluded: set[str],
    is_senior: bool,
    role: str,
    search_query: str,
) -> list[dict]:
    """
    Inject universally relevant anchors + role-specific anchors.
    Runs after diversification so anchors don't consume type slots.
    """
    existing_names = {c["name"].lower() for c in shortlist}
    candidate_map: dict[str, dict] = {c["name"].lower(): c for c in all_candidates}

    # Also search catalog for anchors not in FAISS results
    from catalog_engine import get_catalog_engine
    engine = get_catalog_engine()

    def _get_candidate(name_lower: str) -> dict | None:
        # Check FAISS results first
        c = candidate_map.get(name_lower)
        if c:
            return c
        # Fallback: direct catalog lookup
        entry = engine.get_by_name_exact(name_lower)
        if entry:
            return {
                "name": entry.name, "url": entry.url, "test_type": entry.test_type,
                "keys": entry.keys, "duration": entry.duration,
                "languages": entry.languages, "description": entry.description,
                "score": 0.5,
            }
        return None

    def _try_inject(anchor_name: str):
        if anchor_name in existing_names:
            return
        if any(ex in anchor_name for ex in excluded):
            return
        candidate = _get_candidate(anchor_name)
        if candidate and not _matches_excluded(candidate, excluded):
            if len(shortlist) < 10:
                shortlist.append(candidate)
            else:
                shortlist[-1] = candidate
            existing_names.add(anchor_name)
            logger.debug(f"Anchor injected: {candidate['name']}")

    # Always inject OPQ32r
    _try_inject("occupational personality questionnaire opq32r")

    # Inject Verify G+ for senior roles
    if is_senior:
        _try_inject("shl verify interactive g+")

    # Role-specific anchors
    context = role + " " + search_query
    for keywords, anchor_names in ROLE_ANCHORS:
        if any(kw in context for kw in keywords):
            for anchor_name in anchor_names:
                _try_inject(anchor_name)

    return shortlist


def _skill_overlap_score(candidate: dict, required_skills: list[str]) -> float:
    if not required_skills:
        return 0.0
    text = " ".join([
        candidate.get("name", ""),
        candidate.get("description", ""),
        " ".join(candidate.get("keys", [])),
    ]).lower()
    matches = sum(1 for skill in required_skills if skill in text)
    return matches / len(required_skills)


def _matches_excluded(candidate: dict, excluded: set[str]) -> bool:
    if not excluded:
        return False
    text = " ".join([
        candidate.get("name", ""),
        candidate.get("description", ""),
    ]).lower()
    return any(ex in text for ex in excluded)


def _diversify(scored: list[tuple[float, dict]], type_prefs: list[str]) -> list[dict]:
    type_counts: dict[str, int] = {}
    result = []

    if type_prefs:
        for _, c in scored:
            types = [t.strip() for t in c.get("test_type", "K").split(",")]
            if any(t in type_prefs for t in types):
                primary = types[0]
                if type_counts.get(primary, 0) < MAX_PER_TYPE:
                    result.append(c)
                    type_counts[primary] = type_counts.get(primary, 0) + 1
            if len(result) >= 10:
                break

    seen_names = {c["name"].lower() for c in result}
    for _, c in scored:
        if len(result) >= 10:
            break
        if c["name"].lower() in seen_names:
            continue
        types = [t.strip() for t in c.get("test_type", "K").split(",")]
        primary = types[0]
        if type_counts.get(primary, 0) < MAX_PER_TYPE:
            result.append(c)
            type_counts[primary] = type_counts.get(primary, 0) + 1
            seen_names.add(c["name"].lower())

    return result


def extract_shortlist_from_history(messages: list[dict]) -> list[dict]:
    return []