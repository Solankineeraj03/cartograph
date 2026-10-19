# Milestones

Planning estimate: 10–14 weeks part time, roughly 10–15 focused hours per week; unfamiliar infrastructure and human labeling can extend it. This is a scope estimate, not a delivery promise. Run times must be estimated from pilot throughput on the actual GPU.

| Stage | Indicative effort | Deliverable and completion gate |
|---|---|---|
| 0. Foundation | Created in this repository | CPU retrieval/metric/verifier tests; inspect the synthetic report |
| 1. Real baseline | Week 1 | GPU compatibility probe, locked environments, ESCI ingest, split audit, BM25 and frozen encoder baseline |
| 2. Retrieval research | Weeks 2–3 | Contrastive/graded training, pooling/compression ablations, Qdrant index; per-query held-out report |
| 3. Feature platform | Weeks 4–5 | Ray Data → Iceberg → explicit Feast Parquet export → Redis; late-data/parity/recovery evidence |
| 4. Grounded assistant | Week 6 | End-to-end search, feature fetch, vLLM output, verification, traces; baseline vs SFT |
| 5. Alignment research | Weeks 7–9 | DPO, human preference dataset, reward model, small RLHF and GRPO/RLVR runs; blind held-out evaluation |
| 6. Serving optimization | Weeks 10–11 | Load harness, batching/cache/quantization sweeps, quality-latency-memory report |
| 7. Deployment/governance | Weeks 12–13 | KubeRay, rollback and recovery drills, release manifest, AWS run if budget permits |
| 8. Portfolio release | Week 14 | Reproducible commands, architecture write-up, short demo, experiment report, limitations |

Some stages can overlap, but execute GPU-heavy jobs sequentially. The minimum credible release covers retrieval, feature correctness, SFT/DPO, and measured single-GPU serving. To claim **all requested techniques**, complete reward modeling plus actual RLHF, GRPO/RLVR, the two-GPU experiment, and cloud deployment as well. Do not quietly mark these optional once making that claim.

## Evidence package

Each experiment directory should contain:

```text
config.json
environment-lock.txt
manifest.json
metrics.json
per_example.jsonl
training_or_serving_logs/
figures/
notes.md
```

The manifest binds data checksums/snapshot, split, base model and tokenizer revision, adapter, index, feature/verifier versions, code commit plus dirty status, GPU/driver/CUDA, seed, and exact command. The notes explain the hypothesis, outcome, uncertainty, and failure cases.

For the demo, show one useful search, a constrained bundle, an impossible request, a stale-feature response, a before/after experiment comparison, and a rollback trace. Build the UI only after the pipeline works; a small UI does not need another application platform.

## Interview narrative

Use claims with links to evidence and actual numbers: “Fine-tuned a retrieval encoder on graded product relevance, compared pooling and dimensionality strategies, and measured [actual quality change] at [actual index size].” Describe the synthetic inventory workload explicitly. Never substitute planned results for measured outcomes or claim production traffic for a personal deployment.

Useful engineering discussions: why DPO and online RL were separate branches; why feature availability time matters; why small-model TP sometimes loses; how quantization changes the quality/latency frontier; why reward gains can diverge from independent correctness.

## Next implementation task

On the Linux 3090 host, collect hardware/driver/RAM/disk details, establish separate training and serving environments, pin a small model and encoder, run a 20-step memory pilot and a short generation benchmark, then ingest a bounded real-data slice. Only after those gates choose the final batch/context settings and schedule larger runs.
