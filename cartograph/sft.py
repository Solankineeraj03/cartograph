"""SFT with LoRA on solver-derived demonstrations from train-split bundle tasks only.

Demonstrations: the exact solver's decision, plus a templated explanation computed from the same
evidence (chosen items, total vs budget, one cheaper or higher-scoring alternative, or the
infeasibility reason). Provenance per row: ``label_source = "solver_template_v1"``, so these are
programmatic, not human-written. Loss is on the completion only (TRL prompt-completion format).
``--pilot-steps`` runs a short memory-profiling pilot and records peak memory before full training.
"""

import argparse
import json
import time
from pathlib import Path

from .assistant import messages
from .llm import resolve
from .provenance import Run


def explanation(task: dict) -> str:
    evidence = {e["product_id"]: e for e in task["evidence"]}
    solution = task["solver"]
    if not solution["feasible"]:
        reasons = []
        for reason in solution["reasons"]:
            kind, _, detail = reason.partition(":")
            if kind == "over_budget":
                cheapest = solution.get("cheapest_total_cents")
                reasons.append(f"the cheapest in-stock combination costs {cheapest} cents, above the "
                               f"{task['budget_cents']}-cent budget")
            elif kind == "none_in_stock":
                reasons.append(f"no {detail.replace('_', ' ')} in the evidence is in stock")
            else:
                reasons.append(f"the evidence has no {detail.replace('_', ' ')}")
        return "No valid bundle: " + "; ".join(reasons) + "."
    ids = solution["bundle"]["product_ids"]
    parts = [f"[{i}] {evidence[i]['category'].replace('_', ' ')} at {evidence[i]['price_cents']} cents" for i in ids]
    text = (f"Selected {' and '.join(parts)}; total {solution['bundle']['total_cents']} of "
            f"{task['budget_cents']} cents, all in stock.")
    for i in ids:
        alternatives = [e for e in task["evidence"] if e["category"] == evidence[i]["category"]
                        and e["product_id"] != i and e["in_stock"]]
        cheaper = [e for e in alternatives if e["price_cents"] < evidence[i]["price_cents"]]
        if cheaper:
            c = min(cheaper, key=lambda e: e["price_cents"])
            text += (f" Tradeoff: [{c['product_id']}] is {evidence[i]['price_cents'] - c['price_cents']} cents "
                     f"cheaper but matched the request less closely.")
            break
    return text


def demonstration(task: dict) -> str:
    solution = task["solver"]
    if solution["feasible"]:
        body = {"decision": "bundle", **solution["bundle"]}
    else:
        body = {"decision": "infeasible", "product_ids": [], "total_cents": 0}
    body["explanation"] = explanation(task)
    return json.dumps(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=Path, required=True, help="tasks_train.jsonl")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--pilot-steps", type=int, default=0, help="if set: run only this many steps (memory pilot)")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="smoke test only")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    model_id, revision = resolve(args.model)
    tasks = [json.loads(line) for line in open(args.tasks)]
    if any(t["split"] != "train" for t in tasks):
        raise ValueError("SFT may only use train-split tasks")
    tasks = tasks[:args.max_examples] if args.max_examples else tasks
    rows = [{"prompt": messages(t), "completion": [{"role": "assistant", "content": demonstration(t)}],
             "task_id": t["task_id"], "label_source": "solver_template_v1"} for t in tasks]
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"model_id": model_id, "model_revision": revision, "label_source": "solver_template_v1",
                   "lora": {"r": args.rank, "alpha": 2 * args.rank, "dropout": 0.05, "target_modules": "all-linear"}})
    with Run(args.evidence, "gate_d_sft_pilot" if args.pilot_steps else "gate_d_sft", config, seed=args.seed,
             inputs={"tasks": args.tasks}, lock=Path("envs/training.lock.txt")) as run:
        args.evidence.mkdir(parents=True, exist_ok=True)
        with open(args.evidence / "sft_data_sample.jsonl", "w") as handle:
            for row in rows[:50]:
                handle.write(json.dumps(row) + "\n")
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        lengths = [len(tokenizer.apply_chat_template(r["prompt"] + r["completion"])) for r in rows]
        too_long = sum(n > args.max_length for n in lengths)
        if too_long:
            # Completions sit at the end of each sequence; truncation would drop the supervised answer.
            raise ValueError(f"{too_long} of {len(lengths)} examples exceed max_length={args.max_length} "
                             f"(max {max(lengths)} tokens); raise --max-length")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=torch.float32 if args.cpu else torch.bfloat16)

        class Memory(TrainerCallback):
            def on_log(self, _args, state, control, logs=None, **kw):
                if torch.cuda.is_available() and logs is not None:
                    logs["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
                    logs["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30

        train_config = SFTConfig(
            output_dir=str(args.out), max_length=args.max_length, per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.accum, learning_rate=args.lr, num_train_epochs=args.epochs,
            max_steps=args.pilot_steps or -1, lr_scheduler_type="cosine", warmup_ratio=0.03,
            logging_steps=1 if args.pilot_steps else 10, save_strategy="no", bf16=not args.cpu,
            use_cpu=args.cpu, gradient_checkpointing=args.gradient_checkpointing, report_to=[], seed=args.seed,
            completion_only_loss=True, dataset_num_proc=1)
        trainer = SFTTrainer(model=model, args=train_config, processing_class=tokenizer,
                             train_dataset=Dataset.from_list([{k: r[k] for k in ("prompt", "completion")} for r in rows]),
                             peft_config=LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                                    target_modules="all-linear", task_type="CAUSAL_LM"),
                             callbacks=[Memory()])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = trainer.train()
        seconds = time.perf_counter() - start
        with open(args.evidence / "train_log.jsonl", "w") as handle:
            for entry in trainer.state.log_history:
                handle.write(json.dumps(entry) + "\n")
        if not args.pilot_steps:
            trainer.save_model(str(args.out / "adapter"))
            tokenizer.save_pretrained(str(args.out / "adapter"))
        metrics = {"examples": len(rows), "token_lengths": {"max": max(lengths), "mean": sum(lengths) / len(lengths),
                                                            "over_max_length": too_long},
                   "train_seconds": seconds, "train_loss": result.training_loss, "steps": trainer.state.global_step,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else None,
                   "pilot": bool(args.pilot_steps), "cpu_smoke_test": args.cpu}
        (args.evidence / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        if args.cpu:
            run.manifest["status"] = "smoke_test_cpu"
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
