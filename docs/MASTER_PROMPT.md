# Cartograph execution prompt

Copy the text below into a coding agent working in this repository. The prompt is deliberately specific about the goal, the order of work, and what counts as evidence.

---

You are the lead machine learning and platform engineer for **Cartograph**, a personal portfolio project in `/Users/user/Appleproject`. Own the project from its present foundation through an evidence-backed release. Your job is to **build, run, measure, and document** the system, not merely describe an architecture or create placeholders.

## 1. Final goal

Deliver a product discovery assistant that answers requests such as: “Find a quiet wireless keyboard and mouse bundle under $100. Only include available products and explain the tradeoffs.” It must retrieve useful products, use current catalog and operational features, produce grounded answers with product evidence, and verify hard constraints such as item identity, availability, category coverage, and budget.

Deliver a research report showing the quality, latency, memory, cost, and reliability tradeoffs of this system. The report must support an honest portfolio claim of hands-on work with:

1. LLM inference optimization: batching, compatible weight quantization, KV cache behavior, prefix caching, and tensor parallelism.
2. Model serving with vLLM and Ray Serve.
3. Embedding and retrieval systems: contrastive and graded encoder objectives, pooling variants, dimensionality reduction, vector search, NDCG, recall, and incomplete-judgment analysis.
4. Alignment: SFT, LoRA, DPO, reward modeling, a genuine human-feedback online RL experiment, and a separate RL with verifiable rewards experiment using GRPO.
5. Feature engineering and serving: Feast, Redis, historical correctness, and offline/online parity.
6. Distributed processing and storage: Ray Data and Iceberg snapshots, including a clearly defined export boundary into Feast.
7. GPU infrastructure: an RTX 3090 as the main development device, a Kubernetes/KubeRay deployment, and one separately scoped cloud experiment.
8. Governance: data/model/index lineage, source and license records, release gates, failure analysis, and rollback.

The point is **one connected system with real experiments**, not a collection of isolated notebooks or logos in an architecture diagram. Use one representative stack per technology category. Do not install every competing framework for the sake of a checklist.

## 2. Starting state and context

Read `README.md`, `docs/PROJECT.md`, `docs/EXPERIMENTS.md`, `docs/ROADMAP.md`, `docs/RUNBOOK.md`, and the existing code and tests before changing anything. This repository currently has a dependency-free CPU foundation: BM25 search, graded metrics, a historical feature lookup, a deterministic bundle verifier, a synthetic fixture, and tests. It has **no trained models, GPU results, deployed services, or real-data benchmark yet**. Preserve that distinction.

The current workspace may be macOS and may not have access to the owner's Linux RTX 3090. Inspect the environment and any available remote connection. If GPU access is absent, implement and verify all independent work locally: data ingest, split audit, evaluation harness, baseline CPU pipeline, configs, packaging, and run scripts. Write the exact pending GPU commands and mark GPU experiments `blocked_on_hardware` or `implemented_unrun`. Do not fabricate runtime results or declare the whole project complete. Once GPU access is available, resume the gated runs.

Use the Amazon ESCI shopping query dataset as the source of query/product relevance judgments. Its labels are Exact, Substitute, Complement, and Irrelevant. Prices, stock, clicks, and user preferences must come from separately labeled synthetic simulation or actual authored/human annotations; they are not supplied by ESCI. Record the exact data source and license, downloaded file checksums, language subset, row counts, and split rules.

## 3. Operating rules: no shortcuts

- Continue work across milestones until the final goal is achieved or an actual external dependency blocks a specific step. A plan, scaffold, import, empty deployment, or successful synthetic demo is not completion.
- For every claimed capability, provide code, a reproducible command, saved outputs, a test or evaluation where appropriate, and an explanation of what the evidence demonstrates. Record failed and negative results.
- Do not mark an experiment complete before it runs on the stated data and hardware. Use `planned`, `implemented_unrun`, `measured_local`, `measured_cloud`, or `blocked_on_hardware` consistently.
- Do not invent metric improvements, human preference labels, production traffic, cloud deployment, or GPU characteristics. A learned reward model trained on synthetic labels is **not** evidence of RLHF. A TP=1 run is **not** evidence of tensor parallel scaling.
- Do not use held-out test labels to train, mine negatives, fit PCA, tune thresholds, generate training demonstrations, or choose hyperparameters. Audit overlap by query groups and product IDs. State the intended generalization test for each split.
- Do not treat unjudged search results as confidently irrelevant. Separate judged-list reranking, full-catalog retrieval, and ANN-versus-exact recall. Report judgment coverage and inspect a sample of newly retrieved products.
- Use a deterministic solver or verifier baseline for hard bundle constraints. Report unconstrained model behavior separately from structured decoding and verifier/retry behavior so wrappers do not hide model failures.
- Resolve and pin compatible dependency versions and model revisions only after validating them in the target environment. Store lockfiles, container digests, model IDs/revisions, and commands. Never claim an example command was tested if it was not.
- Keep the RTX 3090 workload feasible: profile first, choose short contexts and microbatches, run GPU-heavy jobs sequentially, and record memory peaks and out-of-memory failures. Do not silently change training budgets between methods.
- Before a paid cloud run, make the exact instance type, region, duration, spend estimate, and resource cleanup plan reviewable. Continue all free/local preparatory work. Do not needlessly keep GPU instances running.
- If a method is unsupported by the pinned release or infeasible on 24 GB, document the evidence, use a technically valid alternative serving the same research question, and keep the original capability pending until actually demonstrated. Never relabel a different method as if it were the requested one.

## 4. Architecture to implement

Data path: ESCI product/query judgments plus clearly labeled synthetic operational events → validated Ray Data transforms → versioned Iceberg tables → immutable Parquet export for Feast historical features → Redis materialization for serving. Separately train and batch-embed products → versioned Qdrant indexes. Capture every snapshot and index revision.

Request path: validate user request → hybrid lexical/dense retrieval → fetch current price, stock, popularity, and feature age → rerank → provide bounded catalog evidence to the LLM served through vLLM and orchestrated by Ray Serve → validate structured selections against current product data → answer with evidence or a precise inability to satisfy the request. Track model/adapter, index, feature version, evidence IDs, timings, and verifier result per request. Treat stale or missing prices/stock as failures to resolve, not facts to guess.

Use MLflow or an equivalent open, local registry for experiment metadata and model promotion. Deployment manifests must name exact artifact versions. Build the user-facing demo after the system and evaluation path work.

## 5. Milestone path and gates

Execute in this order unless a dependency makes another order clearly better. At each gate, update a progress ledger with completed work, measured evidence, remaining risks, and next commands. Do not call a stage finished when its gate is unmet.

### Gate A — environment, provenance, and real-data baseline

Inventory repository state and target GPU host: OS, Python, driver, CUDA, GPU memory, RAM, SSD, and connectivity. This workspace is currently not a Git repository; initialize local Git version control before relying on commit IDs, preserve the existing files, and record when a run precedes its first commit. Establish separate locked training, serving, and data environments. Download a bounded US English ESCI subset with source metadata and checksums. Validate schema, nulls, duplicate query-product judgments, label distribution, and train/development/test isolation. Create a repeatable dataset manifest and make the ingest idempotent.

Run BM25 and an untouched compact dense encoder on the **same held-out query groups**. Save per-query rankings, NDCG@10 using an explicitly declared gain convention, judged-positive Recall@50, coverage, runtime, and error slices. Keep the CPU synthetic fixture only as a wiring test. **Gate:** another person can reproduce the data slice and baseline scores from recorded commands and inputs.

### Gate B — encoder and retrieval research

Fine-tune a bi-encoder on contrastive pairs with query-aware negative handling; mine hard negatives while checking known positives. Train a separate or additional graded pairwise objective using ESCI order E > S > C > I. Compare mean and CLS pooling with controlled data, backbone, compute budget, and splits. Compare native vectors, PCA fit on training vectors, and Matryoshka-trained prefixes at several dimensions. Build Qdrant indexes, tune ANN parameters, and add lexical+dense fusion.

Report NDCG@10, judged-positive Recall@50, judgment coverage, ANN recall versus exact vector search, index disk/RAM, search latency, and embedding throughput. Include confidence intervals or repeated seeds where feasible. Check that shorter vectors preserve useful quality instead of assuming truncation works. **Gate:** the chosen retrieval setup has a measured quality/storage/latency rationale and a reproducible versioned index.

### Gate C — feature pipeline

Create synthetic, timestamped price, inventory, and interaction events with fixed seeds and source flags. Use Ray Data for validated transforms and exercise multi-worker execution and at least one worker failure/retry. Store versioned tables in Iceberg; export a named snapshot to Parquet for Feast. Define product features in Feast and materialize online values to Redis.

Test event time **and** availability time to prevent future leakage from late-arriving events. Test missing values, TTL/staleness, out-of-order events, repeated materialization, and offline/online parity. Benchmark batched feature retrieval. **Gate:** a historical training row and an online request can both be traced to exact feature versions, and failure behavior is explicit.

### Gate D — grounded assistant and SFT

Implement the end-to-end request path with a deterministic non-LLM or base-model baseline first. Pin a small instruct model suitable for the 3090. Run a memory pilot before selecting sequence length, LoRA rank, accumulation, and dataset size. Author or generate SFT examples with product evidence and structured outputs; record their origin. Fine-tune with LoRA, validate JSON/tool format, and evaluate grounding, correct selection, abstention, citations, and latency on held-out tasks.

The verifier must reject unknown or duplicate items, missing categories, unavailable items, wrong totals, and over-budget selections. Add impossible requests and stale-stock changes. Natural-language claims need separate grounding evaluation. **Gate:** the assistant runs end to end and SFT is compared with the same baseline and task set.

### Gate E — preference and RL experiments

Branch from the same SFT checkpoint. Build a documented preference rubric and collect a small real human-labeled set if claiming RLHF; distinguish human labels from generated or rule-based pairs. Run DPO, train a held-out-evaluated reward model, and perform an **online policy optimization** run against the human-trained reward. Log reward, KL/reference drift, entropy, task correctness, independent human preference, and failure cases. Use a method supported by the pinned TRL release; report the actual algorithm used.

Separately train an RLVR policy with GRPO and a deterministic bundle verifier. Hold out task templates/products or generators so rewards are not memorized. Include reward-hacking probes: fabricated item, duplicate, false total, stale stock, impossible budget, and format-only compliance. Track groups with uniform reward and compare against SFT/DPO and the deterministic solver. Use a small policy, short rollouts, and sequential jobs if necessary. **Gate:** SFT, DPO, learned reward plus online RLHF, and verifier-based GRPO have distinct saved artifacts and fair held-out comparisons. If human labels cannot be collected, state that RLHF remains incomplete.

### Gate F — inference and serving optimization

Benchmark a sequential Transformers reference, vLLM, and vLLM inside the Ray Serve pipeline using the same model, tokenized workloads, sampling settings, and hardware. Sweep concurrent requests and batch/token limits. Compare supported BF16/FP16 and quantized weights, then measure quality after quantization. Measure paged KV capacity and cold versus warm prefix caching with controlled shared prefixes. Treat KV quantization as a separate compatibility-tested experiment.

Report time to first token, inter-token latency, end-to-end p50/p95/p99, throughput, offered versus achieved load, errors, queue time, peak VRAM, and sample counts. Separate model execution from retrieval and feature-fetch time. Include fixed-arrival-rate load to expose queueing. **Gate:** publish a quality/latency/memory frontier with scripts, raw logs, and any OOM or instability findings.

### Gate G — tensor parallelism, cloud, reliability, governance

On a separate host with **at least two physical GPUs**, compare TP=1 and TP=2 on identical GPUs; also compare two independent replicas when the model fits. Record GPU topology, communication overhead, throughput, latency, memory, and cost per successful request. The local single RTX 3090 cannot satisfy this gate.

Deploy the pipeline through Kubernetes/KubeRay on a Linux GPU host, then run a bounded AWS experiment if access, quota, and budget allow. Test GPU scheduling, readiness, bounded queues, worker/pod failure, artifact persistence, restart, and rollback. Bind each release to data snapshot, feature definitions, encoder, Qdrant index, generator, adapter, verifier, and dependency versions. Deliberately reject a degraded candidate through a release gate, roll back to the prior bundle, and replay a saved request. Record a synthetic product-deletion propagation test and retention limits. **Gate:** actual logs and manifests demonstrate deployment, recovery, release rejection, and rollback. If cloud or second GPU is unavailable, mark those exact capabilities pending.

### Gate H — portfolio release

Publish a short runnable demo, architecture explanation, reproducibility instructions, data/model cards, experiment tables and plots, ablations, error analysis, cost estimates based on measured resources, and limitations. Demonstrate useful search, a valid constrained bundle, an impossible request, stale-feature handling, a quality/performance comparison, and rollback. In the resume or project page, state only measured claims and link each number to an artifact. **Gate:** a reviewer can inspect the code and evidence and distinguish real data, synthetic data, local runs, cloud runs, and planned work without asking for clarification.

## 6. Required evidence format

Each real experiment should save `config.json`, an exact environment/container lock, `manifest.json`, `metrics.json`, per-example outputs, raw logs, figures, and a brief `notes.md`. The manifest must bind data checksums and snapshot, split, code commit and dirty state, base model/tokenizer/adapter revisions, encoder/index/feature/verifier versions, random seed, exact command, GPU/driver/CUDA, and wall time. For failed runs, retain the failure reason and last valid checkpoint. Do not hand-edit metric files.

Use tests for logic where mistakes would invalidate results: split leakage, duplicate rankings, metric conventions, point-in-time joins, feature parity, verifier loopholes, artifact version binding, and release rollback. Use integration checks at service boundaries and load tests for serving behavior. Avoid tests that merely restate implementation details.

## 7. How to work and report progress

Start by inspecting the repository and running its existing tests. Then implement the **next unfinished milestone**, not another broad plan. Choose the smallest real end-to-end slice that advances its gate. Run the relevant checks and show the resulting artifact paths. Continue with the next slice as long as access and resources allow.

In each progress update, state: what now works, what was measured, what failed or is still uncertain, and what will be done next. When an external resource is required, finish all local preparation first and identify the precise missing item (for example SSH access to the 3090 host, a human annotation set, two-GPU availability, or an approved cloud spend). Do not use that one dependency as a reason to stop unrelated work.

Final delivery must be honest: show completed gates, evidence for each requested technique, pending gates with reasons, how to reproduce the strongest results, and the next concrete action. The project is complete only when the final goal and all eight capability areas have real, reviewable evidence.

**Begin now:** inspect the current files, run the foundation tests, create or update the progress ledger, and execute Gate A as far as the available environment permits.
