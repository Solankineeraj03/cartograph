"""Evaluate a generator on held-out bundle tasks: raw vs wrapped (verify + retry) behavior.

Metrics: strict-JSON rate, format, verified-valid bundles, correct decisions (valid bundle on a
feasible task, abstention on an infeasible one), false abstention, bundles offered for infeasible
tasks, uncited (fabricated) IDs, optimality gap vs the exact solver, and latency. The raw score uses
only the first reply, parsed strictly; wrapped adds lenient parsing and one verifier-feedback retry.
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from .assistant import run_task, user_prompt
from .llm import HFGenerator, OpenAIGenerator, SolverGenerator, VLLMGenerator, resolve, timed
from .provenance import Run


def aggregate(results: list[dict], key: str) -> dict:
    rows = [r[key] for r in results]
    feasible = [r for r in rows if r["feasible"]]
    infeasible = [r for r in rows if not r["feasible"]]
    mean = lambda xs: statistics.fmean(xs) if xs else None  # noqa: E731
    gaps = [r["optimality_gap"] for r in rows if r["optimality_gap"] is not None]
    return {"n": len(rows), "correct": mean([r["correct"] for r in rows]),
            "format_ok": mean([r["format_ok"] for r in rows]),
            "valid_bundle_on_feasible": mean([r["valid_bundle"] for r in feasible]),
            "correct_abstention_on_infeasible": mean([r["decision"] == "infeasible" for r in infeasible]),
            "false_abstention_on_feasible": mean([r["decision"] == "infeasible" for r in feasible]),
            "bundle_offered_on_infeasible": mean([r["decision"] == "bundle" for r in infeasible]),
            "fabricated_id_rate": mean([bool(r["uncited_ids"]) for r in rows]),
            "mean_optimality_gap": mean(gaps), "optimal_rate_on_feasible": mean([g <= 1e-6 for g in gaps]) if gaps else None,
            "error_counts": {e: sum(e in r["errors"] for r in rows) for e in sorted({e for r in rows for e in r["errors"]})}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--engine", required=True, choices=["solver", "hf", "vllm", "openai"])
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    tasks = [json.loads(line) for line in open(args.tasks)]
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    model, revision = resolve(args.model)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"model_id": model, "model_revision": revision, "decoding": "greedy"})
    lock = {"solver": "envs/retrieval.lock.txt", "hf": "envs/training.lock.txt"}.get(args.engine, "envs/serving.lock.txt")
    with Run(args.evidence, f"gate_d_eval_{args.engine}", config, inputs={"tasks": args.tasks},
             lock=Path(lock)) as run:
        if args.engine == "solver":
            generator = SolverGenerator({user_prompt(t): t for t in tasks})
        elif args.engine == "hf":
            generator = HFGenerator(model, revision, args.adapter, args.device, args.max_new_tokens)
        elif args.engine == "vllm":
            generator = VLLMGenerator(model, revision, args.adapter, args.max_new_tokens, quantization=args.quantization)
        else:
            generator = OpenAIGenerator(args.base_url, args.model, args.max_new_tokens)
        generate, calls = timed(generator)
        results = [run_task(task, generate) for task in tasks]
        latency = np.asarray(calls) * 1000
        metrics = {"raw": aggregate(results, "raw"), "wrapped": aggregate(results, "wrapped"),
                   "shown": {k: sum(r["shown"] == k for r in results) for k in
                             ("verified_bundle", "abstain_with_reason", "withheld_unverified")},
                   "generate_calls": len(calls),
                   "latency_ms_per_call": {q: float(np.percentile(latency, int(q[1:]))) for q in ("p50", "p95", "p99")},
                   "latency_note": f"engine={args.engine}, sequential task loop, includes retries"}
        (args.evidence / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        with open(args.evidence / "per_example.jsonl", "w") as handle:
            for r in results:
                handle.write(json.dumps(r) + "\n")
        run.manifest["extra"]["tasks_manifest"] = json.loads((args.tasks.parent / "tasks_manifest.json").read_text()) \
            if (args.tasks.parent / "tasks_manifest.json").exists() else None
        print(json.dumps({"raw": metrics["raw"]["correct"], "wrapped": metrics["wrapped"]["correct"]}))


if __name__ == "__main__":
    main()
