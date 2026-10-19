"""Render held-out task prompts once (chat template applied) for every serving benchmark.

Writes ``prompts.jsonl`` ({"prompt": text}, the format of ``vllm bench serve --dataset-name custom``) and
``prompt_token_ids.json`` (the same prompts tokenized by the pinned tokenizer), so every engine gets identical
inputs. Use ``--custom-skip-chat-template`` with vllm bench, because the template is already applied here.
"""

import argparse
import hashlib
import json
from pathlib import Path

from .assistant import messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tasks = [json.loads(line) for line in open(args.tasks)]
    texts = [tokenizer.apply_chat_template(messages(t), add_generation_prompt=True, tokenize=False) for t in tasks]
    ids = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "prompts.jsonl", "w") as handle:
        for text in texts:
            handle.write(json.dumps({"prompt": text}) + "\n")
    (args.out / "prompt_token_ids.json").write_text(json.dumps(ids))
    lengths = sorted(len(x) for x in ids)
    manifest = {"tasks": str(args.tasks), "tokenizer": args.tokenizer, "n": len(ids),
                "prompt_tokens": {"min": lengths[0], "median": lengths[len(lengths) // 2], "max": lengths[-1]},
                "sha256": hashlib.sha256((args.out / "prompts.jsonl").read_bytes()).hexdigest()}
    (args.out / "prompts_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
