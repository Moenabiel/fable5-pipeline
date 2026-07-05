"""Unit tests for the Fable5 orchestrator pipeline.

These tests use stub/mock backends so they run without any model downloads.
Run with: python -m pytest tests/test_orchestrator.py -v
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference_backends import InferenceResult
from src.orchestrator import Orchestrator
from src.prompts import parse_json_response, build_planner_user, build_drafter_user, build_verifier_user


class StubBackend:
    """Stub backend that returns pre-canned responses for testing."""

    def __init__(self):
        self.call_count = 0

    def generate(self, system_prompt: str, user_prompt: str) -> InferenceResult:
        self.call_count += 1
        if "Planner" in system_prompt or "planner" in system_prompt.lower():
            return InferenceResult(
                text=json.dumps({
                    "steps": ["Step 1: Understand the goal", "Step 2: Research", "Step 3: Synthesize", "Step 4: Verify"],
                    "dod": "Provide a clear, accurate explanation with examples"
                }),
                latency_ms=100,
                model="stub",
                backend="stub",
            )
        elif "Drafter" in system_prompt or "drafter" in system_prompt.lower():
            return InferenceResult(
                text=json.dumps({
                    "candidate_text": "This is a test candidate answer that addresses each step of the plan with intermediate reasoning.",
                    "step_outputs": ["Analyzed the goal", "Researched key concepts", "Synthesized findings"],
                    "sources": [],
                }),
                latency_ms=150,
                model="stub",
                backend="stub",
            )
        elif "Verifier" in system_prompt or "verifier" in system_prompt.lower():
            return InferenceResult(
                text=json.dumps({
                    "factuality": 0.8,
                    "completeness": 0.7,
                    "clarity": 0.9,
                    "issues": [],
                    "reasoning": "Candidate covers all steps and is clear."
                }),
                latency_ms=80,
                model="stub",
                backend="stub",
            )
        return InferenceResult(text="{}", latency_ms=50, model="stub", backend="stub")


def test_parse_json_response():
    assert parse_json_response('{"a": 1}') == {"a": 1}
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('some text {"a": 1} more text') == {"a": 1}
    assert parse_json_response("no json here") is None


def test_parse_json_response_nested():
    text = '{"steps": ["step1", "step2"], "dod": "done"}'
    result = parse_json_response(text)
    assert result is not None
    assert len(result["steps"]) == 2


def test_build_planner_user():
    prompt = build_planner_user("Test goal")
    assert "Test goal" in prompt
    assert "plan" in prompt.lower()


def test_build_drafter_user():
    prompt = build_drafter_user("Test goal", '{"steps": ["s1"]}')
    assert "Test goal" in prompt
    assert "s1" in prompt


def test_build_verifier_user():
    prompt = build_verifier_user("Test goal", '{"steps": []}', '{"candidate_text": "hello"}')
    assert "Test goal" in prompt
    assert "hello" in prompt


def test_orchestrator_plan():
    backend = StubBackend()
    orch = Orchestrator(backend, k_candidates=1)
    plan = orch.plan("Test goal")
    assert "steps" in plan
    assert len(plan["steps"]) == 4
    assert "dod" in plan
    assert plan["dod"] != ""


def test_orchestrator_draft():
    backend = StubBackend()
    orch = Orchestrator(backend, k_candidates=1)
    plan = {"steps": ["s1", "s2"], "dod": "done"}
    candidate = orch.draft("Test goal", plan)
    assert "candidate_text" in candidate
    assert len(candidate["candidate_text"]) > 0


def test_orchestrator_verify():
    backend = StubBackend()
    orch = Orchestrator(backend, k_candidates=1)
    plan = {"steps": ["s1"], "dod": "done"}
    candidate = {"candidate_text": "test answer"}
    score = orch.verify("Test goal", plan, candidate)
    assert "factuality" in score
    assert 0 <= score["factuality"] <= 1


def test_orchestrator_fuse():
    backend = StubBackend()
    orch = Orchestrator(backend, k_candidates=1)
    candidates = [{"candidate_text": "Answer A"}, {"candidate_text": "Answer B"}]
    scores = [{"factuality": 0.9, "completeness": 0.8, "clarity": 0.7}, {"factuality": 0.5, "completeness": 0.4, "clarity": 0.6}]
    result = orch.fuse(candidates, scores)
    assert result == "Answer A"


def test_orchestrator_full_run():
    backend = StubBackend()
    with tempfile.TemporaryDirectory() as tmpdir:
        orch = Orchestrator(backend, k_candidates=2, log_dir=tmpdir)
        trace = orch.run("Test goal")
        assert trace["plan"]["steps"]
        assert len(trace["candidates"]) == 2
        assert len(trace["scores"]) == 2
        assert trace["final_answer"]
        assert trace["metrics"]["total_candidates"] == 2
        assert trace["metrics"]["verifier_pass_rate"] >= 0
        logs = os.listdir(tmpdir)
        assert len(logs) == 1
        with open(os.path.join(tmpdir, logs[0])) as f:
            logged = json.load(f)
        assert logged["goal"] == "Test goal"


def test_orchestrator_error_handling():
    class ErrorBackend:
        def generate(self, system_prompt, user_prompt):
            return InferenceResult(text="", latency_ms=0, model="err", backend="err", error="Connection failed")

    orch = Orchestrator(ErrorBackend(), k_candidates=1)
    trace = orch.run("Test")
    assert trace["plan"].get("error") == "Connection failed"
    assert trace["final_answer"]


if __name__ == "__main__":
    test_parse_json_response()
    test_parse_json_response_nested()
    test_build_planner_user()
    test_build_drafter_user()
    test_build_verifier_user()
    test_orchestrator_plan()
    test_orchestrator_draft()
    test_orchestrator_verify()
    test_orchestrator_fuse()
    test_orchestrator_full_run()
    test_orchestrator_error_handling()
    print("All tests passed!")
