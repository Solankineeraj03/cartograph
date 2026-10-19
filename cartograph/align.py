"""Gate E alignment branches, all starting from the same merged SFT checkpoint.

Subcommands:
  merge   Merge the SFT LoRA adapter into the base weights (the shared branch point).
  pairs   Sample the SFT policy on train tasks and build verifier-labeled preference pairs
          (label_source="verifier_v1": synthetic, NOT human feedback).
  dpo     DPO with a fresh LoRA on a pairs file (verifier or human labels; the source is recorded).
  rm      Train and evaluate a scalar reward model on pairs (pairwise accuracy, length bias).
  grpo    Online GRPO with reward = verifier (RLVR) or = a trained reward model (RLHF only if that
          reward model was trained on human labels; the run refuses otherwise unless --allow-synthetic-rm).
Every run logs reward, KL, entropy (if reported by TRL), and the share of groups with identical rewards.
"""

import argparse
import json
import random
import time
from pathlib import Path

from .assistant import messages, parse, score
from .llm import resolve
from .provenance import Run

HUMAN_SOURCES = {"human_owner_v1"}


def load_tasks(path: Path, split: str) -> list[dict]:
    tasks = [json.loads(line) for line in open(path)]
    if any(t["split"] != split for t in tasks):
        raise ValueError(f"expected only {split}-split tasks in {path}")
    return tasks


def verifier_reward(task: dict, text: str) -> float:
    """1.0 only for a strictly parsed, correct decision. Format alone earns nothing."""
    reply, mode = parse(text)
    return float(mode == "strict" and score(task, reply)["correct"])


def graded_reward(task: dict, text: str) -> float:
    """Predeclared step reward. 0 unless strictly parsed and verifier-correct; 1.0 for a correct abstention or the
    solver-optimal bundle; 0.6 for a valid but suboptimal bundle. Validity always dominates, and choosing a worse valid
    bundle is no longer free (Gate E round 1, binary reward: optimality dropped 0.83 -> 0.59)."""
    reply, mode = parse(text)
    if mode != "strict":
        return 0.0
    result = score(task, reply)
    if not result["correct"]:
        return 0.0
    if not task["solver"]["feasible"]:
        return 1.0
    return 1.0 if (result["optimality_gap"] or 0.0) <= 1e-6 else 0.6


def cmd_merge(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id, revision = resolve(args.model)
    base = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(model_id, revision=revision).save_pretrained(args.out)
    (Path(args.out) / "cartograph_branch.json").write_text(json.dumps(
        {"base": model_id, "base_revision": revision, "sft_adapter": str(args.adapter)}, indent=2))


def cmd_pairs(args):
    from .llm import VLLMGenerator
    tasks = load_tasks(args.tasks, "train")[:args.max_tasks]
    from vllm import SamplingParams
    generator = VLLMGenerator(args.policy, max_new_tokens=256)
    params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=256, n=args.samples, seed=args.seed)
    outputs = generator.llm.chat([messages(t) for t in tasks], params, use_tqdm=False)
    from .sft import demonstration
    pairs, stats = [], {"tasks": len(tasks), "all_correct": 0, "none_correct": 0}
    rng = random.Random(args.seed)
    for task, output in zip(tasks, outputs):
        texts = [o.text for o in output.outputs]
        rewards = [verifier_reward(task, t) for t in texts]
        good = [t for t, r in zip(texts, rewards) if r == 1.0]
        bad = [t for t, r in zip(texts, rewards) if r == 0.0]
        if not bad:
            stats["all_correct"] += 1
            continue
        if not good:
            stats["none_correct"] += 1
            good = [demonstration(task)]  # fall back to the solver demonstration as "chosen"
        pairs.append({"task_id": task["task_id"], "prompt": messages(task),
                      "chosen": [{"role": "assistant", "content": rng.choice(good)}],
                      "rejected": [{"role": "assistant", "content": rng.choice(bad)}],
                      "label_source": "verifier_v1"})
    with open(args.out, "w") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair) + "\n")
    print(json.dumps({**stats, "pairs": len(pairs)}))


def cmd_human_candidates(args):
    """Candidate pairs for owner labeling. Mostly both-valid pairs (answers differ in bundle or explanation; the
    verifier can't rank them, so human judgment adds information), plus ~25% valid-vs-invalid controls that measure
    annotator agreement with the verifier. Train-split tasks only; no labels are produced here."""
    from vllm import SamplingParams

    from .llm import VLLMGenerator
    tasks = load_tasks(args.tasks, "train")[args.offset:args.offset + args.max_tasks]
    generator = VLLMGenerator(args.policy, max_new_tokens=256)
    params = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=256, n=args.samples, seed=args.seed)
    outputs = generator.llm.chat([messages(t) for t in tasks], params, use_tqdm=False)
    rng = random.Random(args.seed)
    both, controls = [], []
    for task, output in zip(tasks, outputs):
        texts = list(dict.fromkeys(o.text.strip() for o in output.outputs))
        valid = [t for t in texts if verifier_reward(task, t) == 1.0]
        invalid = [t for t in texts if verifier_reward(task, t) == 0.0]
        base = {"task_id": task["task_id"], "prompt": messages(task), "label_source": "unlabeled_candidate"}
        if len(valid) >= 2:
            a, b = rng.sample(valid, 2)
            both.append({**base, "pair_type": "both_valid", "chosen": [{"role": "assistant", "content": a}],
                         "rejected": [{"role": "assistant", "content": b}]})
        elif valid and invalid:
            controls.append({**base, "pair_type": "control_valid_vs_invalid",
                             "chosen": [{"role": "assistant", "content": rng.choice(valid)}],
                             "rejected": [{"role": "assistant", "content": rng.choice(invalid)}]})
    n_controls = min(len(controls), max(1, len(both) // 3))
    pairs = both + rng.sample(controls, n_controls)
    rng.shuffle(pairs)
    with open(args.out, "w") as handle:
        for pair in pairs[:args.max_pairs]:
            handle.write(json.dumps(pair) + "\n")
    print(json.dumps({"tasks": len(tasks), "both_valid_available": len(both), "controls_available": len(controls),
                      "written": min(len(pairs), args.max_pairs)}))


def cmd_split_labels(args):
    """Owner labels -> train/test preference pairs. Keeps decided labels from `human_owner_v1` only (ties and skips are
    dropped), and splits by task_id with a hash so no task appears in both files."""
    import hashlib
    rows = [json.loads(line) for line in open(args.labels)]
    decided = [r for r in rows if r.get("label_source") in HUMAN_SOURCES and r.get("choice") in ("a", "b")]
    def is_test(task_id):
        return int(hashlib.sha256(f"rm-split:{task_id}".encode()).hexdigest(), 16) % 100 < args.test_percent
    split = {"train": [r for r in decided if not is_test(r["task_id"])], "test": [r for r in decided if is_test(r["task_id"])]}
    for name, out in (("train", args.train_out), ("test", args.test_out)):
        with open(out, "w") as handle:
            for r in split[name]:
                handle.write(json.dumps({k: r[k] for k in ("task_id", "prompt", "chosen", "rejected", "label_source",
                                                           "pair_type")}) + "\n")
    controls = [r for r in decided if (r.get("pair_type") or "").startswith("control")]
    print(json.dumps({"labels": len(rows), "decided": len(decided), "ties": sum(r.get("choice") == "t" for r in rows),
                      "train": len(split["train"]), "test": len(split["test"]),
                      "control_agreement_with_verifier": (sum(bool(r.get("agrees_with_verifier")) for r in controls) / len(controls))
                      if controls else None}))


def _pairs(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in open(path)]
    sources = {r["label_source"] for r in rows}
    if len(sources) != 1:
        raise ValueError(f"mixed label sources {sources}; train on one source at a time")
    return rows


def cmd_dpo(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer
    rows = _pairs(args.pairs)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"}
    config["label_source"] = rows[0]["label_source"]
    with Run(args.evidence, "gate_e_dpo", config, seed=args.seed, inputs={"pairs": args.pairs},
             lock=Path("envs/training.lock.txt")) as run:
        tokenizer = AutoTokenizer.from_pretrained(args.policy)
        model = AutoModelForCausalLM.from_pretrained(args.policy, dtype=torch.bfloat16)
        lengths = [len(tokenizer.apply_chat_template(r["prompt"] + r[side])) for r in rows for side in ("chosen", "rejected")]
        if max(lengths) > args.max_length:
            # The first DPO run used max_length 1024 on ~1.2-1.35k-token pairs, so TRL truncated them.
            raise ValueError(f"{sum(n > args.max_length for n in lengths)} sequences exceed max_length={args.max_length} "
                             f"(max {max(lengths)})")
        run.manifest["extra"]["sequence_tokens_max"] = max(lengths)
        dataset = Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in rows])
        trainer = DPOTrainer(model=model, ref_model=None, processing_class=tokenizer, train_dataset=dataset,
                             args=DPOConfig(output_dir=str(args.out), beta=args.beta, learning_rate=args.lr,
                                            per_device_train_batch_size=1, gradient_accumulation_steps=args.accum,
                                            num_train_epochs=args.epochs, max_length=args.max_length, bf16=True,
                                            gradient_checkpointing=True, logging_steps=5, save_strategy="no",
                                            report_to=[], seed=args.seed),
                             peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                    target_modules="all-linear", task_type="CAUSAL_LM"))
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        trainer.train()
        trainer.save_model(str(args.out / "adapter"))
        with open(args.evidence / "train_log.jsonl", "w") as handle:
            for entry in trainer.state.log_history:
                handle.write(json.dumps(entry) + "\n")
        run.manifest["extra"].update({"seconds": time.perf_counter() - start, "pairs": len(rows),
                                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30})


def cmd_rm(args):
    import numpy as np
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from trl import RewardConfig, RewardTrainer
    train, test = _pairs(args.pairs), _pairs(args.test_pairs)
    source = train[0]["label_source"]
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"}
    config["label_source"] = source
    with Run(args.evidence, "gate_e_reward_model", config, seed=args.seed,
             inputs={"pairs": args.pairs, "test_pairs": args.test_pairs}, lock=Path("envs/training.lock.txt")) as run:
        model_id, revision = resolve(args.base)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision, num_labels=1,
                                                                   dtype=torch.bfloat16)
        model.config.pad_token_id = tokenizer.pad_token_id
        lengths = [len(tokenizer.apply_chat_template(r["prompt"] + r[side])) for r in train + test
                   for side in ("chosen", "rejected")]
        if max(lengths) > args.max_length:
            # TRL drops over-length pairs silently and the scorer would truncate away the answer (smoke run 1:
            # 1 training step, held-out accuracy 0.15).
            raise ValueError(f"{sum(n > args.max_length for n in lengths)} sequences exceed max_length={args.max_length} "
                             f"(max {max(lengths)})")
        run.manifest["extra"]["sequence_tokens_max"] = max(lengths)
        trainer = RewardTrainer(model=model, processing_class=tokenizer,
                                train_dataset=Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")}
                                                                 for r in train]),
                                args=RewardConfig(output_dir=str(args.out), per_device_train_batch_size=2,
                                                  gradient_accumulation_steps=8, num_train_epochs=args.epochs,
                                                  learning_rate=args.lr, max_length=args.max_length, bf16=True,
                                                  logging_steps=5, save_strategy="no", report_to=[], seed=args.seed),
                                peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear",
                                                       task_type="SEQ_CLS"))
        trainer.train()
        merged = trainer.model.merge_and_unload()  # full model so GRPO can load it as a reward function
        merged.save_pretrained(str(args.out / "reward_model"))
        tokenizer.save_pretrained(str(args.out / "reward_model"))
        model.eval()

        def reward(prompt, reply):
            text = tokenizer.apply_chat_template(prompt + reply, tokenize=False)
            batch = tokenizer(text, return_tensors="pt", truncation=False).to(trainer.model.device)
            with torch.no_grad():
                return float(trainer.model(**batch).logits[0, 0])
        run.manifest["extra"]["train_steps"] = trainer.state.global_step
        rows = []
        for pair in test:
            c, r = reward(pair["prompt"], pair["chosen"]), reward(pair["prompt"], pair["rejected"])
            rows.append({"task_id": pair["task_id"], "chosen": c, "rejected": r, "correct": c > r,
                         "len_chosen": len(pair["chosen"][0]["content"]), "len_rejected": len(pair["rejected"][0]["content"])})
        longer_wins = np.mean([(x["len_chosen"] > x["len_rejected"]) == x["correct"] for x in rows])
        metrics = {"label_source": source, "train_pairs": len(train), "test_pairs": len(test),
                   "pairwise_accuracy": float(np.mean([x["correct"] for x in rows])),
                   "agreement_with_longer_answer": float(longer_wins),
                   "note": "Synthetic verifier labels: this measures pipeline function, not human preference."
                   if source not in HUMAN_SOURCES else "Human-labeled pairs."}
        (args.evidence / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        with open(args.evidence / "per_example.jsonl", "w") as handle:
            for x in rows:
                handle.write(json.dumps(x) + "\n")
        (args.out / "label_source.txt").write_text(source)
        run.manifest["extra"]["metrics"] = metrics


def cmd_grpo(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    tasks = load_tasks(args.tasks, "train")[:args.max_tasks]
    by_id = {t["task_id"]: t for t in tasks}
    group_stats = {"groups": 0, "all_equal": 0}
    reward_kind = args.reward if args.reward in ("verifier", "graded") else "reward_model"
    reward_fn = graded_reward if reward_kind == "graded" else verifier_reward
    if reward_kind == "reward_model":
        source = (Path(args.reward).parent / "label_source.txt").read_text().strip()
        if source not in HUMAN_SOURCES and not args.allow_synthetic_rm:
            raise ValueError(f"reward model labels are {source!r}; RLHF requires human labels")

    samples_path = args.evidence / "completion_samples.jsonl"
    args.evidence.mkdir(parents=True, exist_ok=True)
    calls = {"n": 0}

    def verifier(prompts, completions, task_id, **kwargs):
        texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
        rewards = [reward_fn(by_id[i], text) for i, text in zip(task_id, texts)]
        calls["n"] += 1
        if calls["n"] <= 5 or calls["n"] % 25 == 0:  # keep sampled completions as evidence
            with open(samples_path, "a") as handle:
                handle.write(json.dumps({"call": calls["n"], "task_id": task_id[0], "reward": rewards[0],
                                         "text": texts[0]}) + "\n")
        for start in range(0, len(rewards), args.group):
            group = rewards[start:start + args.group]
            group_stats["groups"] += 1
            group_stats["all_equal"] += len(set(group)) == 1
        return rewards

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"}
    name = {"verifier": "gate_e_grpo_rlvr", "graded": "gate_e_grpo_rlvr_graded"}.get(reward_kind, "gate_e_grpo_rlhf")
    with Run(args.evidence, name, config, seed=args.seed, inputs={"tasks": args.tasks},
             lock=Path("envs/training.lock.txt")) as run:
        tokenizer = AutoTokenizer.from_pretrained(args.policy)
        model = AutoModelForCausalLM.from_pretrained(args.policy, dtype=torch.bfloat16)
        reward_funcs = verifier
        if reward_kind == "reward_model":  # only reached with human labels (or --allow-synthetic-rm for tests)
            from transformers import AutoModelForSequenceClassification
            reward_funcs = AutoModelForSequenceClassification.from_pretrained(args.reward, num_labels=1,
                                                                              dtype=torch.bfloat16)
        prompt_lengths = [len(tokenizer.apply_chat_template(messages(t), add_generation_prompt=True)) for t in tasks]
        if max(prompt_lengths) > args.max_prompt_length:
            # TRL truncates long prompts, dropping evidence or the system prompt (gate_e smoke run 1 had all-zero reward).
            raise ValueError(f"{sum(n > args.max_prompt_length for n in prompt_lengths)} prompts exceed "
                             f"max_prompt_length={args.max_prompt_length} (max {max(prompt_lengths)})")
        run.manifest["extra"]["prompt_tokens_max"] = max(prompt_lengths)
        dataset = Dataset.from_list([{"prompt": messages(t), "task_id": t["task_id"]} for t in tasks])
        trainer = GRPOTrainer(model=model, reward_funcs=reward_funcs, processing_class=tokenizer, train_dataset=dataset,
                              args=GRPOConfig(output_dir=str(args.out), num_generations=args.group,
                                              per_device_train_batch_size=args.group, gradient_accumulation_steps=args.accum,
                                              max_prompt_length=args.max_prompt_length, max_completion_length=args.max_completion_length, beta=args.beta,
                                              learning_rate=args.lr, max_steps=args.steps, bf16=True,
                                              gradient_checkpointing=True, logging_steps=1, save_strategy="no",
                                              temperature=args.temperature, report_to=[], seed=args.seed,
                                              log_completions=False),
                              peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                     target_modules="all-linear", task_type="CAUSAL_LM"))
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        trainer.train()
        trainer.save_model(str(args.out / "adapter"))
        with open(args.evidence / "train_log.jsonl", "w") as handle:
            for entry in trainer.state.log_history:
                handle.write(json.dumps(entry) + "\n")
        run.manifest["extra"].update({"seconds": time.perf_counter() - start, "group_stats": group_stats,
                                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("merge"); p.set_defaults(func=cmd_merge)  # noqa: E702
    p.add_argument("--model", default="qwen2.5-1.5b-instruct"); p.add_argument("--adapter", required=True)  # noqa: E702
    p.add_argument("--out", required=True)
    p = sub.add_parser("pairs"); p.set_defaults(func=cmd_pairs)  # noqa: E702
    for a in ("--policy", "--out"):
        p.add_argument(a, required=True)
    p.add_argument("--tasks", type=Path, required=True); p.add_argument("--max-tasks", type=int, default=1500)  # noqa: E702
    p.add_argument("--samples", type=int, default=4); p.add_argument("--seed", type=int, default=13)  # noqa: E702
    p = sub.add_parser("human_candidates"); p.set_defaults(func=cmd_human_candidates)  # noqa: E702
    p.add_argument("--policy", required=True); p.add_argument("--out", required=True)  # noqa: E702
    p.add_argument("--tasks", type=Path, required=True); p.add_argument("--offset", type=int, default=1500)  # noqa: E702
    p.add_argument("--max-tasks", type=int, default=1500); p.add_argument("--samples", type=int, default=6)  # noqa: E702
    p.add_argument("--temperature", type=float, default=1.0); p.add_argument("--max-pairs", type=int, default=500)  # noqa: E702
    p.add_argument("--seed", type=int, default=17)
    p = sub.add_parser("split_labels"); p.set_defaults(func=cmd_split_labels)  # noqa: E702
    p.add_argument("--labels", type=Path, required=True); p.add_argument("--train-out", type=Path, required=True)  # noqa: E702
    p.add_argument("--test-out", type=Path, required=True); p.add_argument("--test-percent", type=int, default=20)  # noqa: E702
    p = sub.add_parser("dpo"); p.set_defaults(func=cmd_dpo)  # noqa: E702
    p.add_argument("--policy", required=True); p.add_argument("--pairs", type=Path, required=True)  # noqa: E702
    p.add_argument("--out", type=Path, required=True); p.add_argument("--evidence", type=Path, required=True)  # noqa: E702
    p.add_argument("--beta", type=float, default=0.1); p.add_argument("--lr", type=float, default=5e-6)  # noqa: E702
    p.add_argument("--accum", type=int, default=16); p.add_argument("--epochs", type=float, default=1.0)  # noqa: E702
    p.add_argument("--seed", type=int, default=13); p.add_argument("--max-length", type=int, default=1600)  # noqa: E702
    p = sub.add_parser("rm"); p.set_defaults(func=cmd_rm)  # noqa: E702
    p.add_argument("--base", default="qwen2.5-0.5b-instruct"); p.add_argument("--pairs", type=Path, required=True)  # noqa: E702
    p.add_argument("--test-pairs", type=Path, required=True); p.add_argument("--out", type=Path, required=True)  # noqa: E702
    p.add_argument("--evidence", type=Path, required=True); p.add_argument("--epochs", type=float, default=1.0)  # noqa: E702
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--seed", type=int, default=13)  # noqa: E702
    p.add_argument("--max-length", type=int, default=1600)
    p = sub.add_parser("grpo"); p.set_defaults(func=cmd_grpo)  # noqa: E702
    p.add_argument("--policy", required=True); p.add_argument("--tasks", type=Path, required=True)  # noqa: E702
    p.add_argument("--reward", default="verifier", help="'verifier', 'graded', or a reward model directory")
    p.add_argument("--allow-synthetic-rm", action="store_true", help="pipeline test only; never reported as RLHF")
    p.add_argument("--out", type=Path, required=True); p.add_argument("--evidence", type=Path, required=True)  # noqa: E702
    p.add_argument("--max-tasks", type=int, default=2000); p.add_argument("--group", type=int, default=4)  # noqa: E702
    p.add_argument("--accum", type=int, default=4); p.add_argument("--steps", type=int, default=300)  # noqa: E702
    p.add_argument("--beta", type=float, default=0.04); p.add_argument("--lr", type=float, default=1e-5)  # noqa: E702
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-prompt-length", type=int, default=1280)
    p.add_argument("--max-completion-length", type=int, default=384)
    p.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
