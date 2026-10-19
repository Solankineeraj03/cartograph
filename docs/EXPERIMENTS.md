# Experiments and evaluation protocol

All experiments below are planned. No improvement numbers have been measured.

## Shared protocol

Freeze the dataset split, model revision, tokenizer/template, dependencies, random seeds, and hardware manifest. Save per-query/per-request outputs rather than only aggregate scores. Use development data for selection; run the held-out test after selecting a configuration. Report failed runs and OOMs. For small training experiments use three seeds where feasible and paired bootstrap confidence intervals across queries/tasks. Treat repeated queries as groups, not independent rows.

## A. Retrieval

| ID | Experiment | Comparison |
|---|---|---|
| R0 | BM25 | Reproducible lexical baseline |
| R1 | Frozen dense encoder | Dense vs lexical vs reciprocal-rank fusion |
| R2 | Contrastive fine-tuning | Exact positives and mined irrelevant negatives; control false negatives |
| R3 | Graded ranking | Within-query pairs with E > S > C > I; weighted logistic pairwise loss |
| R4 | Pooling | Mean vs CLS with the same backbone, data, and training budget |
| R5 | Compression | Native dimensions vs train-fitted PCA vs Matryoshka-trained dimensions |
| R6 | Feature reranking | Retrieval scores alone vs current stock, price match, and popularity features |
| R7 | ANN tuning | Qdrant approximate search vs exact nearest neighbors with identical vectors |

For R2, a batch must not accidentally mark a query's other valid products as negatives. Use query-aware sampling, known-positive masking, and audit mined negatives. For R3 compare the graded objective separately, then test combining it with contrastive loss. Example loss: `w * log(1 + exp(-(s(query, higher) - s(query, lower))))`, where `w` is a predeclared relevance-gap weight. Do not rank products from unrelated queries against one another using raw grade values.

Primary metrics: NDCG@10 with **direct gains E=1, S=0.1, C=0.01, I=0**; judged-positive Recall@50 with E/S relevant; report the exact convention with every run. Use the official benchmark evaluator when claiming benchmark-comparable results, and verify agreement on fixtures. Record all-zero relevance queries separately. The local foundation uses this explicit direct-gain convention.

ESCI contains incomplete judgments. Separate **ranking within judged candidate sets** from **retrieval across the full catalog**. Report full-catalog judged-positive recall, judged coverage, and a manually judged sample of new results; do not interpret all unjudged products as truly irrelevant. ANN recall is agreement with exact vector search, a different quantity from relevance recall.

For 384-dimensional embeddings compare 384/256/128/64 dimensions. Naive truncation is a control, not an assumed valid compression method. Matryoshka training optimizes several prefixes; PCA is fit only on training vectors. Compression primarily reduces vector storage/search work, not encoder forward-pass cost. [Sentence Transformers explanation](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html).

At one million vectors, float32 coordinate storage is `1,000,000 × dimension × 4` bytes: 1.536 GB at 384 dimensions and 0.512 GB at 128 dimensions. This is arithmetic for raw coordinates only; report actual database disk/RAM with graph and metadata overhead.

Suggested acceptance hypotheses: tuned retrieval beats the strongest frozen baseline on held-out NDCG; a compressed index retains at least 98% of that NDCG while reducing measured index size. Negative findings are valid results.

## B. Alignment

Branch experiments from a shared SFT checkpoint so comparisons remain interpretable:

```text
Instruct baseline → SFT (LoRA)
                      ├─ DPO on preferences
                      ├─ learned reward → online RLHF experiment
                      └─ deterministic verifier → GRPO / RLVR experiment
```

SFT, DPO, GRPO, and reward modeling have dedicated TRL workflows. Resolve the supported trainer APIs for the pinned release before implementation; don't assume older PPO examples match current TRL. [TRL documentation](https://huggingface.co/docs/trl/index).

1. **SFT:** start with 2,000–5,000 examples, catalog evidence in context, 1,024-token total sequence cap, rank-8/16 LoRA, microbatch one, gradient accumulation. Supervise answer/tool output rather than prompt tokens. Validate answer format, task correctness, and grounding separately.
2. **DPO:** start with 500–1,500 pairs, clearly record human versus synthetic labels. Cache reference log probabilities when supported. Tune beta on development preferences; preserve answer length distributions to avoid shortcuts.
3. **Reward model:** score answer pairs conditioned on the exact prompt and evidence. Hold out query/product groups. Measure pairwise accuracy, length bias, and performance on subtle invalid answers. Start with a small encoder/scalar head to fit the hardware budget.
4. **RLHF:** personally annotate a manageable seed set (for example 300–500 pairs); retain the rubric and annotation records. Optimize a small policy online against this human-trained reward using a supported algorithm such as GRPO or RLOO with reference-policy regularization. Report KL, entropy, reward, independent human preference, and task correctness. Training a reward model alone does not complete this milestone. If every label is synthetic, describe it as synthetic-preference alignment rather than human-feedback evidence.
5. **RLVR with GRPO:** use catalog tasks with a unique checkable feasibility condition; start with four short completions per prompt and profile memory. Track groups where all rewards are equal, because they offer little relative signal. Use a separate branch from SFT and a separate held-out task generator. Add mixed learned/verifiable rewards only as a later ablation.

The structured bundle output is `{"product_ids": [...], "total_cents": integer}`. The verifier rejects unknown IDs, duplicates, unavailable items, missing categories, wrong totals, and over-budget bundles. In the foundation, success is binary. Future shaping must never let format-only rewards outweigh invalid product selections. Natural-language grounding requires additional claim/evidence checks and human evaluation; valid JSON does not establish factual prose.

Add held-out adversarial tasks: impossible budget; nonexistent item; stale stock; duplicate item; falsely cheap total; fake citation; missing required category. For abstention, verify infeasibility with a bounded exact solver on the small synthetic task catalog. Do not award correctness simply for refusing. Compare learning to a deterministic constraint solver, which is an important non-LLM baseline.

Evaluate unconstrained model outputs to measure what training learned, then separately evaluate production-style structured decoding and verifier/retry wrappers. Otherwise decoding constraints can hide training failures. Report success with and without retrieval errors, unsupported-claim rate, valid citation rate, blind pairwise preference, and task latency.

## C. Serving

The factorial space is large; begin with one-factor ablations and combine only promising settings.

| Axis | Initial values |
|---|---|
| Engine | Transformers sequential reference; vLLM; vLLM behind Ray Serve |
| Weight format | BF16/FP16 baseline; verified Ampere-compatible 4-bit checkpoint |
| Concurrency | 1, 2, 4, 8, 16, stopping at failure/SLO violation |
| Prompt/output tokens | 256/64, 1024/128, 2048/256 |
| Prefix reuse | 0%, 50%, 90%; cold and warm cache |
| Batch limits | several max-sequence and token-budget settings |
| Parallelism | TP=1 local; TP=1 and TP=2 on one separate multi-GPU host |

Measure time to first token (TTFT), inter-token latency, end-to-end p50/p95/p99, input/output tokens per second, requests per second, errors, queue delay, peak VRAM, and energy if available. Count actual tokenizer output; don't substitute character counts. Report offered and achieved load, including failures. Include a fixed-arrival-rate load test to expose queueing, plus a concurrency sweep. Use enough requests for meaningful tail percentiles and report sample counts.

Use identical tokenized prompts, output budgets, sampling settings, and hardware for comparisons. Run quality checks after quantization. Separate model execution from HTTP/Ray overhead and retrieval/feature latency. Measure cold startup separately from warmed steady state. Prefix caching reuses prefill work; it does not remove decode work or cache arbitrary similar text. [vLLM prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/).

Paged KV allocation, prefix reuse, and KV quantization are different mechanisms. Use BF16/FP16 KV initially. Treat quantized KV as an additional hardware/backend-dependent experiment, not a guaranteed 3090 optimization. Verify the exact chosen quantization kernel against the pinned release. [vLLM quantization matrix](https://docs.vllm.ai/en/v0.22.1/features/quantization/).

Before the load sweep, declare a provisional SLO such as p95 TTFT ≤1 second and error rate <1% for the 1024/128 workload. It is a target to test, not an achieved claim. If unattainable, publish the observed frontier. Prefix hit rates must reflect genuine repeated system/catalog prefixes, not duplicated evaluation answers.

Cloud TP experiment: compare one GPU, two GPUs with TP, and (if the model fits) two independent replicas on the same hardware. Log GPU model, PCIe/NVLink topology, collective communication overhead, and cost per successful request at the selected latency bound. Two-way TP is not guaranteed to be faster for small models.

## D. Feature platform, reliability, and governance

- Replay out-of-order and late-arriving events; confirm historical rows never access future information.
- Compare Redis serving values to their exact offline materialization snapshot, including types, defaults, and TTL behavior.
- Measure p95 batched feature-fetch latency and stale/missing feature rates. Verify behavior under Redis outage rather than silently inventing prices.
- Kill a Ray preprocessing worker and verify idempotent retries without duplicate products/events; run at least one multi-node CPU job before claiming distributed operation.
- Kill a serving pod, replay load, and record recovery/error behavior. Demonstrate bounded backpressure and cancellation.
- Reject a deliberately degraded model through a release gate, then replay a request after rollback of the model/index/feature bundle.
- Preserve charts, raw outputs, manifests, and negative results in a release artifact.
