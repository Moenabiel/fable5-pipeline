"""Fable5 Orchestrator — planner → drafter(s) → verifier → fuser.

Implements the core multi-step reasoning pipeline:
1. Plan: Convert user goal into numbered steps + Definition of Done
2. Draft: Generate k candidate answers following the plan
3. Verify: Score candidates on factuality, completeness, clarity
4. Fuse: Merge best candidates into polished final answer
5. Log: Structured JSON logs of all prompts, responses, and scores

Usage:
    python -m src.orchestrator --prompt "Your question here" --backend llama_cpp --model-path model.bin
    python -m src.orchestrator --prompt "Your question here" --backend hf_bnb --model-id mistralai/Mistral-7B-Instruct-v0.3
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

from .inference_backends import create_backend, InferenceResult
from .prompts import (
    PLANNER_SYSTEM,
    DRAFTER_SYSTEM,
    VERIFIER_SYSTEM,
    build_planner_user,
    build_drafter_user,
    build_verifier_user,
    parse_json_response,
)


class Orchestrator:
    """Multi-step reasoning orchestrator: plan → draft → verify → fuse."""

    def __init__(self, backend, k_candidates: int = 3, log_dir: str = "logs"):
        self.backend = backend
        self.k = k_candidates
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def plan(self, goal: str) -> dict:
        """Step 1: Convert user goal into numbered plan + Definition of Done."""
        user_prompt = build_planner_user(goal)
        result = self.backend.generate(PLANNER_SYSTEM, user_prompt)
        if result.error:
            return {"steps": [], "dod": "", "error": result.error}
        parsed = parse_json_response(result.text)
        if parsed is None:
            return {"steps": [result.text], "dod": "Complete the task"}
        return {
            "steps": parsed.get("steps", []),
            "dod": parsed.get("dod", "Complete the task"),
            "_latency_ms": result.latency_ms,
        }

    def draft(self, goal: str, plan: dict, candidate_index: int = 0) -> dict:
        """Step 2: Generate one candidate answer following the plan."""
        plan_json = json.dumps(plan, indent=2)
        user_prompt = build_drafter_user(goal, plan_json)
        result = self.backend.generate(DRAFTER_SYSTEM, user_prompt)
        if result.error:
            return {"candidate_text": "", "step_outputs": [], "sources": [], "error": result.error}
        parsed = parse_json_response(result.text)
        if parsed is None:
            return {
                "candidate_text": result.text,
                "step_outputs": [],
                "sources": [],
                "_latency_ms": result.latency_ms,
            }
        parsed["_latency_ms"] = result.latency_ms
        return parsed

    def verify(self, goal: str, plan: dict, candidate: dict) -> dict:
        """Step 3: Score a candidate on factuality, completeness, clarity."""
        plan_json = json.dumps(plan, indent=2)
        candidate_json = json.dumps(
            {k: v for k, v in candidate.items() if not k.startswith("_")},
            indent=2,
        )
        user_prompt = build_verifier_user(goal, plan_json, candidate_json)
        result = self.backend.generate(VERIFIER_SYSTEM, user_prompt)
        if result.error:
            return {"factuality": 0, "completeness": 0, "clarity": 0, "issues": [result.error]}
        parsed = parse_json_response(result.text)
        if parsed is None:
            return {
                "factuality": 0.5,
                "completeness": 0.5,
                "clarity": 0.5,
                "issues": ["Could not parse verifier output"],
                "_raw": result.text,
            }
        parsed["_latency_ms"] = result.latency_ms
        return parsed

    def fuse(self, candidates: list[dict], scores: list[dict]) -> str:
        """Step 4: Select the best candidate based on verifier scores."""
        if not candidates:
            return "No candidates generated."
        scored = []
        for cand, score in zip(candidates, scores):
            avg = (
                score.get("factuality", 0) * 0.5
                + score.get("completeness", 0) * 0.3
                + score.get("clarity", 0) * 0.2
            )
            scored.append((avg, cand))
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        text = best.get("candidate_text", "")
        if not text:
            text = json.dumps(best, indent=2)
        return text

    def run(self, goal: str) -> dict:
        """Execute the full pipeline: plan → draft → verify → fuse."""
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        trace = {
            "run_id": run_id,
            "goal": goal,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "plan": {},
            "candidates": [],
            "scores": [],
            "final_answer": "",
            "metrics": {},
        }

        # Step 1: Plan
        plan = self.plan(goal)
        trace["plan"] = plan

        # Step 2: Draft k candidates
        candidates = []
        for i in range(self.k):
            cand = self.draft(goal, plan, candidate_index=i)
            candidates.append(cand)
        trace["candidates"] = [
            {k: v for k, v in c.items() if not k.startswith("_")} for c in candidates
        ]

        # Step 3: Verify each candidate
        scores = []
        for cand in candidates:
            score = self.verify(goal, plan, cand)
            scores.append(score)
        trace["scores"] = [
            {k: v for k, v in s.items() if not k.startswith("_")} for s in scores
        ]

        # Step 4: Fuse
        final = self.fuse(candidates, scores)
        trace["final_answer"] = final

        # Metrics
        latencies = [
            c.get("_latency_ms", 0) for c in candidates
        ] + [s.get("_latency_ms", 0) for s in scores]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        pass_count = sum(
            1 for s in scores
            if s.get("factuality", 0) >= 0.6 and s.get("completeness", 0) >= 0.6
        )
        trace["metrics"] = {
            "total_candidates": self.k,
            "verifier_pass_rate": pass_count / self.k if self.k > 0 else 0,
            "avg_latency_ms": round(avg_latency, 1),
            "plan_steps": len(plan.get("steps", [])),
            "total_time_ms": 0,
        }

        trace["finished_at"] = datetime.now(timezone.utc).isoformat()

        # Log
        log_path = os.path.join(self.log_dir, f"run_{run_id}.json")
        with open(log_path, "w") as f:
            json.dump(trace, f, indent=2)

        return trace


def main():
    parser = argparse.ArgumentParser(description="Fable5 Orchestrator Pipeline")
    parser.add_argument("--prompt", required=True, help="User goal/prompt")
    parser.add_argument("--backend", choices=["llama_cpp", "hf_bnb"], default="llama_cpp")
    parser.add_argument("--model-path", help="Path to GGUF model (llama_cpp backend)")
    parser.add_argument("--model-id", help="HuggingFace model ID (hf_bnb backend)")
    parser.add_argument("--k", type=int, default=3, help="Number of candidates")
    parser.add_argument("--log-dir", default="logs", help="Directory for JSON logs")
    parser.add_argument("--llama-cpp-bin", default="./llama.cpp/main", help="Path to llama.cpp binary")
    args = parser.parse_args()

    if args.backend == "llama_cpp" and not args.model_path:
        parser.error("--model-path required for llama_cpp backend")
    if args.backend == "hf_bnb" and not args.model_id:
        parser.error("--model-id required for hf_bnb backend")

    backend_kwargs = {}
    if args.backend == "llama_cpp":
        backend_kwargs = {"model_path": args.model_path, "llama_cpp_bin": args.llama_cpp_bin}
    elif args.backend == "hf_bnb":
        backend_kwargs = {"model_id": args.model_id}

    backend = create_backend(args.backend, **backend_kwargs)
    orch = Orchestrator(backend, k_candidates=args.k, log_dir=args.log_dir)

    print(f"Running Fable5 pipeline ({args.backend})...")
    print(f"Goal: {args.prompt}")
    print(f"Candidates: {args.k}")
    print("-" * 60)

    trace = orch.run(args.prompt)

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(trace["final_answer"])
    print("\n" + "=" * 60)
    print("METRICS:")
    print(json.dumps(trace["metrics"], indent=2))
    print(f"\nFull trace logged to: {args.log_dir}/run_{trace['run_id']}.json")


if __name__ == "__main__":
    main()
