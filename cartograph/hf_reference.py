"""Sequential Transformers reference: one prompt at a time, greedy, fixed output length.

Uses the token IDs from ``serving_prompts`` and forces exactly ``--output-len`` new tokens
(min_new_tokens = max_new_tokens), matching ``vllm bench serve --ignore-eos`` output lengths.
Time to first token is measured with a streamer; peak GPU memory comes from the CUDA allocator.
"""

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True, help="prompt_token_ids.json")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    prompts = json.loads(args.prompts.read_text())[:args.num_prompts]
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for i, ids in enumerate(prompts):
        input_ids = torch.tensor([ids], device="cuda")
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        kwargs = dict(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), do_sample=False,
                      max_new_tokens=args.output_len, min_new_tokens=args.output_len,
                      pad_token_id=tokenizer.eos_token_id, streamer=streamer)
        torch.cuda.synchronize()
        start = time.perf_counter()
        thread = threading.Thread(target=model.generate, kwargs=kwargs)
        thread.start()
        first = None
        for chunk in streamer:
            if first is None and chunk:
                first = time.perf_counter()
        thread.join()
        torch.cuda.synchronize()
        end = time.perf_counter()
        rows.append({"i": i, "prompt_tokens": len(ids), "ttft_ms": (first - start) * 1000,
                     "e2e_ms": (end - start) * 1000, "tpot_ms": (end - first) * 1000 / (args.output_len - 1)})
    ttft, e2e, tpot = (np.array([r[k] for r in rows]) for k in ("ttft_ms", "e2e_ms", "tpot_ms"))
    total = e2e.sum() / 1000
    metrics = {"engine": "transformers_sequential", "n": len(rows), "output_len": args.output_len,
               **{f"{name}_{q}": float(np.percentile(v, int(q[1:]))) for name, v in
                  (("ttft_ms", ttft), ("e2e_ms", e2e), ("tpot_ms", tpot)) for q in ("p50", "p95", "p99")},
               "request_throughput": len(rows) / total, "output_throughput": len(rows) * args.output_len / total,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with open(args.out / "per_example.jsonl", "w") as handle:
        for r in rows:
            handle.write(json.dumps(r) + "\n")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
