"""Blind pairwise preference labeling for RLHF (owner-annotated). Rubric: docs/PREFERENCE_RUBRIC.md.

Each item shows a request, its catalog evidence, and two answers in random order (A/B). The
annotator picks A, B, or tie (or skips). Records keep annotator, rubric version, timestamp, and
the unblinded mapping, with label_source="human_owner_v1". Labels are appended and resumable.
"""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

RUBRIC_VERSION = "pref-rubric-v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True, help="pairs jsonl from `align pairs` (unlabeled use)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    done = {json.loads(l)["task_id"] for l in open(args.out)} if args.out.exists() else set()
    rng = random.Random(args.seed)
    items = [json.loads(l) for l in open(args.candidates)]
    for item in items:
        if item["task_id"] in done:
            continue
        answers = [("x", item["chosen"][0]["content"]), ("y", item["rejected"][0]["content"])]
        rng.shuffle(answers)
        print("\n" + "=" * 100 + f"\n{item['task_id']}\n" + item["prompt"][1]["content"])
        for label, (_, text) in zip("AB", answers):
            print(f"\n--- Answer {label} ---\n{text}")
        choice = ""
        while choice not in {"a", "b", "t", "s", "q"}:
            choice = input("\nPrefer [a]/[b], [t]ie, [s]kip, [q]uit: ").strip().lower()
        if choice == "q":
            break
        if choice == "s":
            continue
        record = {"task_id": item["task_id"], "prompt": item["prompt"], "label_source": "human_owner_v1",
                  "annotator": args.annotator, "rubric_version": RUBRIC_VERSION,
                  "labeled_at_utc": datetime.now(timezone.utc).isoformat(), "choice": choice,
                  "shown_order": [k for k, _ in answers], "pair_type": item.get("pair_type", "unknown")}
        if choice != "t":
            preferred = answers[0] if choice == "a" else answers[1]
            other = answers[1] if choice == "a" else answers[0]
            record.update(chosen=[{"role": "assistant", "content": preferred[1]}],
                          rejected=[{"role": "assistant", "content": other[1]}],
                          agrees_with_verifier=(preferred[0] == "x") if item.get("pair_type", "").startswith("control") else None)
        with open(args.out, "a") as handle:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
