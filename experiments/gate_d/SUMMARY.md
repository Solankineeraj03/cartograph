# Gate D: grounded bundle assistant, baseline vs SFT

Status: `measured_local` on GPU 1 of gpu-host (GPU 0 excluded after two Xid 79 events).
Generator: Qwen2.5-1.5B-Instruct `989aa798`, vLLM 0.11.0 offline engine, greedy decoding, max 256 new tokens.

## Tasks (`tasks_manifest.json`)
Bundle tasks over the synthetic operational catalog. Candidates come from BM25 over titles and the derived category;
prices and stock come from the Gate C `ingest_2` export at each task's as-of time. The catalog is split into two disjoint
product halves by hash: 3,000 train tasks (seed 101) and 500 test tasks (seed 202), with **0 shared products**.
About 80% are feasible; budgets are drawn around the exact solver's optimum. Limitation: each split uses only ~450
distinct products, because evidence retrieval is deterministic per (category, adjective).

## Scoring (`cartograph/assistant.py`)
**Raw** = the first reply, parsed strictly as exactly one JSON object. **Wrapped** = lenient JSON extraction plus one
retry with the verifier's errors. A correct answer is a verified bundle on a feasible task (one in-stock product per
required category, no duplicates, IDs from the evidence, exact total, within budget), or an abstention on a task the
exact solver proves infeasible. Abstaining on a feasible task is wrong.

## Results (500 held-out tasks)

| | Solver | Base raw | Base wrapped | SFT raw | SFT wrapped |
|---|---|---|---|---|---|
| Correct | 1.000 | 0.000 | 0.190 | **0.688** | **0.748** |
| Strict-format replies | 1.000 | 0.000 | — | 1.000 | — |
| Valid bundle, feasible tasks (n=403) | 1.000 | 0.000 | 0.000 | 0.660 | 0.692 |
| Correct abstention, infeasible (n=97) | 1.000 | 0.000 | 0.979 | 0.804 | 0.979 |
| False abstention, feasible | 0 | — | 0.983 | 0.032 | 0.203 |
| Bundle offered when infeasible | 0 | — | 0 | 0.196 | 0.021 |
| Fabricated product ID rate | 0 | 0 | 0 | 0.008 | 0.010 |
| Optimal among valid bundles | 1.000 | — | — | 0.838 | 0.814 |
| Latency per call p50 / p95 | — | 0.43 / 0.85 s | | 1.03 / 1.09 s | |

Answers shown to a user after verification (SFT wrapped): 279 verified bundles, 177 refusals with a reason,
44 withheld because they couldn't be verified. **No unverified bundle is ever shown.**

## Reading
- The base model wraps JSON in Markdown code fences (0% strict) and, once parsed, refuses almost every task.
  Its 19% "correct" is just the infeasible share. It never produced a valid bundle.
- SFT teaches the format (100% strict) and real selection: 66% valid bundles on unseen products, 84% of them optimal.
- **Remaining errors are mostly arithmetic and constraint checking.** Raw errors: 103 over budget (the model writes
  "total 5398 of 3300 cents" and still offers the bundle), 46 wrong totals, 9 out of stock, 4 unknown IDs.
- **The retry wrapper trades errors for refusals.** It raises correctness 0.688 → 0.748 but raises false abstention
  0.032 → 0.203: after negative feedback the model often gives up rather than fixing the bundle. That's why raw and
  wrapped are reported separately.
- The exact solver solves this task completely in milliseconds. The LLM's value is the natural-language explanation
  and request handling; constraint satisfaction belongs to the verifier and solver.

## Training (sft_full)
LoRA r=16, alpha 32, all linear layers, lr 2e-4 cosine, 2 epochs, microbatch 1 × accumulation 16 (376 steps), bf16,
gradient checkpointing (the pilot reserved 23.0 GiB without it), max length 1,536 (0 examples truncated). Loss
0.81 → 0.013. 2,165 s. Peak memory 5.55 GiB allocated / 22.8 GiB reserved. Demonstrations come from the exact solver
plus a templated explanation (`label_source = solver_template_v1`), not human writing.
Adapter SHA-256 `63d343e8…24c` (`sft_full/adapter_sha256.txt`).

## Failed or stopped runs kept
`failed_sft_pilot_len1024_truncation` and `failed_sft_full_len1024_killed`: at max length 1,024, 2,928 of 3,000
examples would have lost their supervised answer to truncation. The run was stopped, and `sft.py` now refuses to
train when any example would be truncated.

## Not yet measured for Gate D
The served request path (Ray Serve with features from Redis), citation checks for claims inside the explanation text,
and a structured-decoding (JSON-schema-constrained) variant.
