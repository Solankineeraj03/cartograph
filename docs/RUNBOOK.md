# Execution guide

## Environments

The current development machine is macOS. Its CPU foundation has no third-party dependencies. GPU training/serving is intended for a Linux host with the RTX 3090; there is no configured GPU connection in this repository.

Start with 32 GB host RAM and generous SSD space as planning assumptions; 64 GB RAM and approximately 100 GB free disk give more room for dataset/model versions. Inventory the actual machine before selecting data scale. Avoid launching the entire eventual platform during a training run.

On the GPU host capture:

```bash
nvidia-smi
python3 --version
df -h
free -h
```

Create separate environments for training, serving, and data/platform dependencies. vLLM and training stacks may require different Torch/CUDA versions. At milestone 1 choose compatible released versions, run import/model-load smoke checks, then commit exact lockfiles and container digests. This repository intentionally does not invent an untested GPU dependency lock. Save immutable Hugging Face revisions rather than relying on moving model names alone.

The first runtime check must confirm CUDA availability, BF16 support for the selected stack, actual free VRAM, tokenizer loading, and 20 training steps without OOM. Record peak allocated/reserved memory. A 24 GB device does not imply 24 GB is free for model weights.

## Initial GPU budgets

These are settings to probe, not measured memory guarantees:

| Workload | Starting settings |
|---|---|
| Encoder tuning | ≤256 tokens; batch 16; lower batch before reducing experimental controls |
| 1.5B SFT | ≤1024 total tokens; microbatch 1; accumulation 16; LoRA rank 8/16 |
| 1.5B DPO | ≤1024 prompt+completion; microbatch 1; precomputed reference log-probs if supported |
| Small online RL | 0.5B–1.5B; ≤512 prompt and 128 completion; group 4; sequential scoring |
| vLLM service | 2048 total context; max 4 sequences; ~80% GPU memory allocation; TP=1 |

Stop the serving engine before training. If OOM occurs, reduce context/completion length, rollout group/microbatch, and model size in that order as appropriate; record the deviation. Check algorithm divisibility requirements when changing GRPO groups/batches. Do not run simultaneous training and an independent full-memory vLLM process by default.

Illustrative initial serving command, to verify against the locked vLLM release and resolved local model revision:

```bash
vllm serve /path/to/pinned-model-snapshot \
  --served-model-name cartograph \
  --host 127.0.0.1 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.80 \
  --tensor-parallel-size 1 \
  --enable-prefix-caching
```

This command is a runbook example, not a tested launch script. vLLM performs generation scheduling; Ray Serve adds request orchestration and service scaling. Avoid an extra HTTP hop where a supported Ray Serve LLM integration can own the vLLM engine directly. [Ray Serve LLM documentation](https://docs.ray.io/en/latest/serve/llm/index.html).

## Platform rollout

1. Local processes for initial data and retrieval research.
2. Containers for Qdrant, Redis, and MLflow; persist named data volumes.
3. Iceberg catalog/object storage and Ray Data after the baseline is reproducible. Start with local storage/REST catalog; export exact snapshots to Feast-readable files.
4. KubeRay on Linux, GPU device plugin, explicit GPU requests, readiness probes, resource limits, and persistent artifacts. A single-GPU node gets one GPU-serving replica; CPU pipeline workers scale separately.
5. One AWS deployment with object storage, private networking, scoped identity, and externalized artifacts. Keep GPU workers ephemeral. Do not leave cloud infrastructure running for a portfolio screenshot.

The cloud milestone needs an account, available GPU quota, and a chosen spending cap. No cloud resources have been created. Estimate cost from current regional on-demand rates immediately before that run: `instance_hours × hourly_rate + storage + transfer`. For TP use a multi-GPU instance and compare TP=1/2 there, not a cloud GPU against the local 3090. GCP is a follow-on portability exercise; do not claim it from an AWS-only deployment.

## Report discipline

The CPU demo's metrics validate wiring only. It uses a handful of authored products and queries, so even perfect scores say nothing about model performance. Its hashes are a provenance foundation, not tamper-proof auditing. GPU runs require a richer manifest with actual driver, CUDA, model and environment details.

Keep results statuses explicit: `planned`, `implemented_unrun`, `measured_local`, or `measured_cloud`. Publish only observed claims. Save failure reports as well as successful runs.
