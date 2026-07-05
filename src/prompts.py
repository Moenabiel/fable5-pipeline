"""Prompt templates for the Fable5 orchestrator pipeline.

Each component (Planner, Drafter, Verifier) has a system prompt template
and a user prompt builder. These are designed to elicit chain-of-thought
reasoning, step-by-step planning, and structured JSON output.
"""

import json
from typing import Optional


PLANNER_SYSTEM = (
    "You are 'Planner', an instruction-following planning agent. Given a user goal, produce:\n"
    "- A numbered list of at most 6 discrete steps to reach the goal.\n"
    "- A short 'Definition of Done' (one or two sentences) describing the criteria for success.\n"
    "- Output strictly in JSON: {\"steps\": [\"...\"], \"dod\": \"...\" }.\n"
    "Follow the KISS principle; be explicit about required artifacts and verification steps."
)

DRAFTER_SYSTEM = (
    "You are 'Drafter'. Given a plan (JSON) and user goal, produce one complete candidate answer that:\n"
    "- Addresses each plan step in order, showing intermediate reasoning in short bullet points.\n"
    "- Provides a clear final answer (3-8 sentences).\n"
    "- If external facts are used, include a 'sources' array with exact citations.\n"
    "- Output JSON: {\"candidate_text\":\"...\",\"step_outputs\":[...],\"sources\":[...]}."
)

VERIFIER_SYSTEM = (
    "You are 'Verifier'. Given a candidate answer and the original plan, evaluate:\n"
    "- factuality: 0.0-1.0 (no hallucinations -> 1.0)\n"
    "- completeness: 0.0-1.0 (covers plan steps)\n"
    "- clarity: 0.0-1.0\n"
    "- issues: list of short issue descriptions\n"
    "Provide concise reasoning for each score. Output JSON: "
    "{\"factuality\":0.0,\"completeness\":0.0,\"clarity\":0.0,\"issues\":[\"...\"],\"reasoning\":\"...\"}."
)


def build_planner_user(goal: str) -> str:
    """Build the user prompt for the Planner component."""
    return f"Goal: {goal}\n\nProduce a numbered plan (max 6 steps) and a Definition of Done. Output JSON."


def build_drafter_user(goal: str, plan_json: str) -> str:
    """Build the user prompt for a Drafter component."""
    return (
        f"Goal: {goal}\n\n"
        f"Plan:\n{plan_json}\n\n"
        "Produce a complete candidate answer following the plan. Output JSON."
    )


def build_verifier_user(goal: str, plan_json: str, candidate_json: str) -> str:
    """Build the user prompt for the Verifier component."""
    return (
        f"Goal: {goal}\n\n"
        f"Plan:\n{plan_json}\n\n"
        f"Candidate:\n{candidate_json}\n\n"
        "Evaluate factuality, completeness, clarity, and list issues. Output JSON."
    )


def parse_json_response(text: str) -> Optional[dict]:
    """Extract and parse JSON from model output, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return None
