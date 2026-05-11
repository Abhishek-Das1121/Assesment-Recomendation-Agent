"""
test_traces.py — Local evaluator trace replay.
"""

import argparse
import sys
import requests
import time
from dataclasses import dataclass, field

BASE_URL = "http://127.0.0.1:8000"

# Seconds to wait between turns — prevents per-minute quota exhaustion.
# Groq Free Tier is ~30 RPM. 2 calls per turn = 15 turns per minute max.
# 4 seconds ensures we stay well within the limit.
TURN_DELAY_SECONDS = 4

@dataclass
class TraceResult:
    trace_id: str
    turns: int = 0
    schema_valid: bool = True
    turn_limit_honored: bool = True
    final_recommendations: list[dict] = field(default_factory=list)
    expected_assessments: list[str] = field(default_factory=list)
    recall_at_10: float = 0.0
    errors: list[str] = field(default_factory=list)

TRACES = {
    "C1": {
        "description": "Senior leadership / CXO selection",
        "messages": [
            {"role": "user", "content": "We need a solution for senior leadership."},
            {"role": "user", "content": "The pool consists of CXOs, director-level positions; people with more than 15 years of experience."},
            {"role": "user", "content": "Selection — comparing candidates against a leadership benchmark."},
            {"role": "user", "content": "Perfect, that's what we need."},
        ],
        "expected_names": [
            "Occupational Personality Questionnaire OPQ32r",
            "OPQ Universal Competency Report 2.0",
            "OPQ Leadership Report",
        ],
    },
    "C2": {
        "description": "Senior Rust engineer, no Rust test exists",
        "messages": [
            {"role": "user", "content": "I'm hiring a senior Rust engineer for high-performance networking infrastructure. What assessments should I use?"},
            {"role": "user", "content": "Yes, go ahead. Should I also add a cognitive test for this level?"},
            {"role": "user", "content": "That works. Thanks."},
        ],
        "expected_names": [
            "Smart Interview Live Coding",
            "SHL Verify Interactive G+",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    "C3": {
        "description": "500 entry-level contact centre agents",
        "messages": [
            {"role": "user", "content": "We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus. What should we use?"},
            {"role": "user", "content": "English."},
            {"role": "user", "content": "US accent."},
            {"role": "user", "content": "Perfect — new simulation for volume, old solution for finalists. Confirmed."},
        ],
        "expected_names": [
            "SVAR Spoken English (US) (New)",
            "Contact Center Call Simulation (New)",
            "Entry Level Customer Serv - Retail & Contact Center",
            "Customer Service Phone Simulation",
        ],
    },
    "C9": {
        "description": "Senior full-stack / backend Java engineer",
        "messages": [
            {"role": "user", "content": "Here's the JD for a senior engineer role: Senior Full-Stack Engineer, 5+ years, Core Java, Spring, REST APIs, Angular, SQL, AWS, Docker. Backend-leaning. Day-one priorities are Core Java and Spring; SQL is constant."},
            {"role": "user", "content": "Senior IC. They lead design on their own services but don't manage other engineers directly."},
            {"role": "user", "content": "Add AWS and Docker. Drop REST — the API design signal will already come through in Spring and the live interview."},
            {"role": "user", "content": "Keep Verify G+. Locking it in."},
        ],
        "expected_names": [
            "Core Java (Advanced Level) (New)",
            "Spring (New)",
            "SQL (New)",
            "Amazon Web Services (AWS) Development (New)",
            "Docker (New)",
            "SHL Verify Interactive G+",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    "C4": {
        "description": "Graduate financial analysts — numerical + finance knowledge",
        "messages": [
            {"role": "user", "content": "Hiring graduate financial analysts — final-year students, no work experience. We need numerical reasoning and a finance knowledge test."},
            {"role": "user", "content": "Good. Can you also add a situational judgement element — work-context decision making for graduates?"},
            {"role": "user", "content": "That covers it. Numerical + Graduate Scenarios as first filter, domain tests for shortlisted candidates."},
        ],
        "expected_names": [
            "SHL Verify Interactive - Numerical Reasoning",
            "Financial Accounting (New)",
            "Graduate Scenarios",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    "C6": {
        "description": "Plant operators — chemical facility, safety critical",
        "messages": [
            {"role": "user", "content": "We're hiring plant operators for a chemical facility. Safety is absolute top priority — reliability, procedure compliance, never cutting corners. What do you recommend?"},
            {"role": "user", "content": "We're industrial. The 8.0 bundle is the right fit. Confirmed."},
        ],
        "expected_names": [
            "Dependability and Safety Instrument (DSI)",
            "Workplace Health and Safety (New)",
        ],
    },
}

def chat(messages: list[dict]) -> dict:
    resp = requests.post(
        f"{BASE_URL}/chat",
        json={"messages": messages},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

def validate_schema(response: dict) -> list[str]:
    errors = []
    if "reply" not in response:
        errors.append("Missing 'reply' field")
    if "recommendations" not in response:
        errors.append("Missing 'recommendations' field")
    elif not isinstance(response["recommendations"], list):
        errors.append("'recommendations' must be a list")
    else:
        for i, rec in enumerate(response["recommendations"]):
            for field_name in ("name", "url", "test_type"):
                if field_name not in rec:
                    errors.append(f"recommendations[{i}] missing '{field_name}'")
            # Added .lower() check for more robust URL validation
            if not rec.get("url", "").lower().startswith("https://www.shl.com"):
                errors.append(f"recommendations[{i}] invalid URL: {rec.get('url')}")
    if "end_of_conversation" not in response:
        errors.append("Missing 'end_of_conversation' field")
    return errors

def compute_recall_at_10(final_recs: list[dict], expected_names: list[str]) -> float:
    if not expected_names:
        return 1.0
    rec_names_lower = {r["name"].lower() for r in final_recs[:10]}
    found = sum(
        1 for e in expected_names
        if any(e.lower() in rn or rn in e.lower() for rn in rec_names_lower)
    )
    return found / len(expected_names)

def format_assistant_message(response: dict) -> str:
    content = response.get("reply", "")
    recs = response.get("recommendations", [])
    if recs:
        content += "\n\nRecommendations:\n"
        for r in recs:
            content += f"- {r['name']} ({r['test_type']}): {r['url']}\n"
    return content

def run_trace(trace_id: str, trace: dict) -> TraceResult:
    result = TraceResult(trace_id=trace_id, expected_assessments=trace["expected_names"])
    messages = []
    final_recs = []

    print(f"\n{'='*60}")
    print(f"Trace: {trace_id} — {trace['description']}")
    print(f"{'='*60}")

    for user_msg in trace["messages"]:
        messages.append(user_msg)
        result.turns += 1

        if result.turns > 8:
            result.turn_limit_honored = False
            result.errors.append(f"Turn limit exceeded at turn {result.turns}")
            break

        try:
            response = chat(messages)
        except Exception as e:
            result.errors.append(f"Turn {result.turns} request failed: {e}")
            break

        schema_errors = validate_schema(response)
        if schema_errors:
            result.schema_valid = False
            result.errors.extend(schema_errors)

        recs = response.get("recommendations", [])
        if recs:
            final_recs = recs

        print(f"\nTurn {result.turns}:")
        print(f"  User: {user_msg['content'][:80]}")
        print(f"  Reply: {response.get('reply', '')[:100]}")
        print(f"  Recs: {len(recs)} | eoc: {response.get('end_of_conversation')}")

        messages.append({
            "role": "assistant",
            "content": format_assistant_message(response),
        })

        if response.get("end_of_conversation"):
            break

        # Critical for Groq Free Tier
        time.sleep(TURN_DELAY_SECONDS)

    result.final_recommendations = final_recs
    result.recall_at_10 = compute_recall_at_10(final_recs, trace["expected_names"])
    return result

def main():
    global BASE_URL
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", help="Run a specific trace (e.g. C9)")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    BASE_URL = args.url

    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Health check: {resp.json()}")
    except Exception as e:
        print(f"Server not reachable at {BASE_URL}: {e}")
        sys.exit(1)

    traces_to_run = {args.trace: TRACES[args.trace]} if args.trace else TRACES

    results = []
    for trace_id, trace in traces_to_run.items():
        result = run_trace(trace_id, trace)
        results.append(result)
        # Fixed 10s sleep for inter-trace recovery
        if len(traces_to_run) > 1:
            print(f"\nPausing 10s for quota recovery...")
            time.sleep(10)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    mean_recall = sum(r.recall_at_10 for r in results) / len(results) if results else 0
    for r in results:
        status = "✓" if r.schema_valid and r.turn_limit_honored else "✗"
        print(f"  {status} {r.trace_id}: Recall@10={r.recall_at_10:.2f}, turns={r.turns}")
    print(f"\nMean Recall@10: {mean_recall:.3f}")

if __name__ == "__main__":
    main()