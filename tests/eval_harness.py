"""Evaluation harness for the Fable5 pipeline.

Runs automated checks on orchestrator output:
- Structural checks (plan present, JSON valid, steps non-empty)
- Verifier pass rate
- Hallucination heuristic detection
- Candidate diversity
- Latency metrics

Usage:
    python -m tests.eval_harness --prompts tests/sample_prompts.jsonl --backend llama_cpp --model-path model.bin
    python -m tests.eval_harness --prompts tests/sample_prompts.jsonl --backend hf_bnb --model-id mistralai/Mistral-7B-Instruct-v0.3
"""

import argparse
import json
import os
import re
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator import Orchestrator
from src.inference_backends import create_backend


def check_plan_structure(plan: dict) -> dict:
    """Check that plan has valid structure."""
    issues = []
    steps = plan.get("steps", [])
    dod = plan.get("dod", "")
    if not steps:
        issues.append("No steps in plan")
    if len(steps) > 6:
        issues.append(f"Plan has {len(steps)} steps (max 6)")
    if not dod:
        issues.append("No Definition of Done")
    return {"pass": len(issues) == 0, "issues": issues}


def check_json_output(text: str) -> bool:
    """Check if output contains valid JSON."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                json.loads(text[start:end])
                return True
            except json.JSONDecodeError:
                pass
        return False


def check_hallucination_heuristic(text: str) -> dict:
    """Simple heuristic hallucination check.

    Flags text that:
    - Contains "I don't know" or "I'm not sure" (uncertainty markers)
    - Contains very specific numbers without context (potential fabrication)
    - Is extremely short (potential incomplete answer)
    """
    issues = []
    if len(text) < 20:
        issues.append("Very short answer — potential incomplete response")
    uncertainty = ["i don't know", "i'm not sure", "i cannot", "i can't verify"]
    for marker in uncertainty:
        if marker in text.lower():
            issues.append(f"Uncertainty marker: '{marker}'")
    specific_numbers = re.findall(r'\b\d{4,}\b', text)
    if len(specific_numbers) > 3:
        issues.append(f"Many specific numbers ({len(specific_numbers)}) — verify factuality")
    return {"pass": len(issues) == 0, "issues": issues}


def check_candidate_diversity(candidates: list[dict]) -> dict:
    """Check that candidates are meaningfully different."""
    texts = [c.get("candidate_text", "") for c in candidates]
    if len(texts) < 2:
        return {"pass": False, "issues": ["Need at least 2 candidates for diversity check"]}
    unique_lengths = set(len(t) for t in texts)
    if len(unique_lengths) == 1:
        return {"pass": False, "issues": ["All candidates have identical length — low diversity"]}
    char_overlap = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            shorter = min(len(texts[i]), len(texts[j]))
            if shorter == 0:
                continue
            overlap = sum(1 for a, b in zip(texts[i], texts[j]) if a == b) / shorter
            char_overlap.append(overlap)
    avg_overlap = sum(char_overlap) / len(char_overlap) if char_overlap else 0
    if avg_overlap > 0.8:
        return {"pass": False, "issues": [f"Average character overlap: {avg_overlap:.2f} (>0.8 — candidates too similar)"]}
    return {"pass": True, "issues": []}


def evaluate_run(trace: dict) -> dict:
    """Evaluate a single orchestrator run."""
    results = {
        "run_id": trace.get("run_id", "unknown"),
        "checks": {},
        "pass_count": 0,
        "total_checks": 0,
    }

    plan_check = check_plan_structure(trace.get("plan", {}))
    results["checks"]["plan_structure"] = plan_check
    results["total_checks"] += 1
    if plan_check["pass"]:
        results["pass_count"] += 1

    json_check = check_json_output(json.dumps(trace.get("plan", {})))
    results["checks"]["json_valid"] = {"pass": json_check, "issues": [] if json_check else ["Plan not valid JSON"]}
    results["total_checks"] += 1
    if json_check:
        results["pass_count"] += 1

    scores = trace.get("scores", [])
    pass_rate = trace.get("metrics", {}).get("verifier_pass_rate", 0)
    results["checks"]["verifier_pass_rate"] = {
        "pass": pass_rate >= 0.6,
        "issues": [] if pass_rate >= 0.6 else [f"Pass rate {pass_rate:.2f} < 0.6"],
        "value": pass_rate,
    }
    results["total_checks"] += 1
    if pass_rate >= 0.6:
        results["pass_count"] += 1

    candidates = trace.get("candidates", [])
    diversity = check_candidate_diversity(candidates)
    results["checks"]["candidate_diversity"] = diversity
    results["total_checks"] += 1
    if diversity["pass"]:
        results["pass_count"] += 1

    final = trace.get("final_answer", "")
    hallucination = check_hallucination_heuristic(final)
    results["checks"]["hallucination_heuristic"] = hallucination
    results["total_checks"] += 1
    if hallucination["pass"]:
        results["pass_count"] += 1

    results["pass_rate"] = results["pass_count"] / results["total_checks"] if results["total_checks"] > 0 else 0
    return results


def run_evaluation(prompts_path: str, backend_type: str, model_kwargs: dict, k: int = 3, log_dir: str = "logs"):
    """Run evaluation on a set of prompts."""
    with open(prompts_path) as f:
        prompts = [json.loads(line) for line in f if line.strip()]

    backend = create_backend(backend_type, **model_kwargs)
    orch = Orchestrator(backend, k_candidates=k, log_dir=log_dir)

    all_results = []
    for i, item in enumerate(prompts):
        prompt = item.get("prompt", item.get("goal", ""))
        print(f"\n[{i+1}/{len(prompts)}] Evaluating: {prompt[:80]}...")
        trace = orch.run(prompt)
        eval_result = evaluate_run(trace)
        all_results.append(eval_result)
        print(f"  Pass rate: {eval_result['pass_rate']:.2f} ({eval_result['pass_count']}/{eval_result['total_checks']})")

    overall_pass = sum(r["pass_rate"] for r in all_results) / len(all_results) if all_results else 0
    summary = {
        "total_prompts": len(prompts),
        "overall_pass_rate": round(overall_pass, 3),
        "per_prompt": all_results,
    }

    summary_path = os.path.join(log_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Overall pass rate: {overall_pass:.2%}")
    print(f"Results saved to: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Fable5 Evaluation Harness")
    parser.add_argument("--prompts", required=True, help="Path to JSONL prompts file")
    parser.add_argument("--backend", choices=["llama_cpp", "hf_bnb"], default="llama_cpp")
    parser.add_argument("--model-path", help="Path to GGUF model")
    parser.add_argument("--model-id", help="HuggingFace model ID")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    model_kwargs = {}
    if args.backend == "llama_cpp":
        if not args.model_path:
            parser.error("--model-path required for llama_cpp")
        model_kwargs = {"model_path": args.model_path}
    else:
        if not args.model_id:
            parser.error("--model-id required for hf_bnb")
        model_kwargs = {"model_id": args.model_id}

    run_evaluation(args.prompts, args.backend, model_kwargs, args.k, args.log_dir)


if __name__ == "__main__":
    main()
