# Gate F: serving optimization on one RTX 3090 (GPU 1)

Status: `measured_local`. Model: the merged SFT checkpoint (Qwen2.5-1.5B-Instruct + SFT LoRA) that the assistant
serves. Workload: the 500 held-out task prompts (chat template applied once; 895–1,200 tokens, median 1,045), exactly
128 output tokens (`--ignore-eos`), greedy. Every vLLM measurement uses vLLM 0.11.0's own `vllm bench serve` against
`vllm serve` (max model length 2,048, GPU memory utilization 0.80). GPU memory, power, and utilization sampled every
0.5 s (`*/gpu_samples.csv`). No request failed in any run.

## Engines (bf16)

| Engine | Concurrency | Requests/s | Output tok/s | TTFT p50 / p95 | TPOT p50 | E2E p95 | Peak GPU memory |
|---|---|---|---|---|---|---|---|
| Transformers, sequential (n=100) | 1 | 0.42 | 54 | 121 / 128 ms | 17.8 ms | 2.41 s | 3.3 GiB reserved |
| vLLM | 1 | 1.44 | 184 | 34 / 59 ms | 5.2 ms | 0.72 s | 20.4 GiB (pre-allocated) |
| vLLM | 8 | 9.02 | 1,154 | 24 / 35 ms | 6.6 ms | 0.87 s | |
| vLLM | 16 | 15.5 | 1,980 | 29 / 56 ms | 7.2 ms | 0.96 s | |
| vLLM | 32 | 22.4 | 2,869 | 59 / 910 ms | 8.4 ms | 2.23 s | |
| vLLM | 64 | 22.0 | 2,817 | 147 / 1,871 ms | 14.9 ms | 5.85 s | |

Open-loop Poisson arrivals (300 requests each), achieved vs offered: 2 → 1.99, 4 → 3.96, 8 → 7.83, 16 → 15.2,
32 → 28.5 requests/s. At an offered 32 requests/s, TTFT p95 is still 58 ms but TPOT doubles to 14 ms (E2E p95 2.16 s).

- vLLM decodes 3.4× faster per token than sequential Transformers at concurrency 1 (5.2 vs 17.8 ms TPOT) and reaches
  about 53× the throughput at saturation (2,869 vs 54 tok/s).
- Closed-loop throughput saturates at about 22 requests/s from concurrency 32. Beyond that, extra concurrency only adds latency.
- **Caveat:** the sweep reuses the same prompts on one server, so after warmup most prefixes are cached. Those TTFTs are
  warm-cache numbers; see the prefix experiment below for cold vs warm.

## Batch limits (`--max-num-seqs`, offered concurrency 32)

| max-num-seqs | Requests/s | TTFT p50 | TPOT p50 | E2E p95 |
|---|---|---|---|---|
| 4 | 4.31 | 6,707 ms (queueing) | 6.8 ms | 7.63 s |
| 16 | 10.70 | 1,681 ms | 11.6 ms | 3.31 s |
| default (256) | 22.4 | 59 ms | 8.4 ms | 2.23 s |

Capping concurrent sequences moves waiting from decode into the queue (TTFT), and throughput drops in proportion.

## KV cache and prefix caching

KV capacity at 0.80 utilization, 2,048-token context: **544,496 tokens** (14.5 GiB, enough for 265 full-length
requests) for bf16 weights (2.89 GiB), 589,424 with FP8 weights (1.68 GiB), and 609,344 with bnb 4-bit (1.15 GiB).

| Server | Workload | TTFT p50 / p95 | E2E p50 | Output tok/s |
|---|---|---|---|---|
| prefix cache on | first pass, 100 task prompts, concurrency 1 (cold) | 57 / 63 ms | 716 ms | 178 |
| prefix cache on | same prompts again (warm) | **16 / 19 ms** | 676 ms | 188 |
| prefix cache off | first pass | 65 / 68 ms | 722 ms | 177 |
| prefix cache off | second pass | 64 / 66 ms | 721 ms | 178 |
| prefix cache on | shared 1,024-token prefix (4 prefixes) + 128 unique, concurrency 8 | 74 / 110 ms | 476 ms | 1,061 |
| prefix cache off | same | 280 / 292 ms | 845 ms | 606 |

A warm cache cuts TTFT 3.6× (57 → 16 ms) on repeated task prompts. It doesn't change decode time, so E2E drops only 6%.
With genuinely shared long prefixes under concurrency, caching cuts TTFT 3.8× and raises throughput 1.75×.

## Weight quantization (same merged model, no calibration data)

| Weights | Weight memory | Conc. 1: tok/s, TPOT | Conc. 8: tok/s, TPOT | Conc. 32: tok/s, TTFT p95 | Correct raw (Δ vs bf16, 95% CI) | Correct wrapped (Δ, CI) |
|---|---|---|---|---|---|---|
| bf16 | 2.89 GiB | 184, 5.2 ms | 1,154, 6.6 ms | 2,869, 910 ms | 0.690 | 0.748 |
| FP8 weight-only (Marlin, Ampere) | 1.68 GiB | **227, 4.1 ms** | **1,598, 4.7 ms** | **2,987**, 1,037 ms | 0.676 (−0.014 [−0.032, +0.002]) | 0.730 (−0.018 [−0.036, −0.002]) |
| bitsandbytes 4-bit | 1.15 GiB | 219, 4.1 ms | 683, 11.2 ms | 1,956, 1,133 ms | **0.590 (−0.100 [−0.138, −0.060])** | 0.732 (−0.016 [−0.046, +0.014]) |

- **FP8 weight-only is the best point on the frontier for this model:** 42% less weight memory, 23–38% faster decode at
  low and medium concurrency, and +4% saturated throughput, for a small quality cost (−1.4 to −1.8 points).
- **bnb 4-bit saves the most memory but is slower under batching** (TPOT 11.2 ms at concurrency 8 vs 6.6 bf16) and
  loses 10 points of raw correctness. The verify-and-retry wrapper hides most of that loss (−1.6, not significant),
  which is why raw and wrapped quality are reported separately.
- The KV memory freed by smaller weights is irrelevant here: even bf16 leaves room for 265 concurrent 2,048-token
  requests, far more than throughput saturation needs (concurrency ~32).
- Quality is measured on the bundle task with the vLLM offline engine; latency is measured on the server.

## Provisional SLO check
Declared target: p95 TTFT ≤ 1 s and error rate < 1% for this ~1k-input / 128-output workload. bf16 meets it up to
concurrency 16 (56 ms) and borderline at 32 (910 ms); it fails at 64 (1.87 s). FP8 fails at 32 (1.04 s). Errors were 0 in all runs.

## Not measured here
Two-GPU tensor parallelism (GPU 0 excluded after two Xid 79 events; see `docs/CLOUD_PLAN.md`). Energy per request:
power is sampled in `gpu_samples.csv` but not yet integrated per configuration. The Ray Serve comparison is below.

## Ray Serve request path (`ray_serve/`)
Ray Serve 2.49.2 with two deployments: a GPU `Generator` (in-process vLLM engine, same merged model and engine settings)
and a CPU `Cartograph` ingress (validation, BM25 retrieval, Feast online features from Redis, exact solver as
feasibility oracle, LLM answer, verifier, one retry, full trace). Release `2406781e60ea`; features materialized from
`ingest_2` with watermark 2026-10-01T02:00Z.

**Overhead vs bare vLLM** (same prompts, `vllm bench serve` against the Serve `/v1/completions` endpoint):

| Concurrency | Bare vLLM: req/s, TTFT p50, TPOT p50 | Ray Serve: req/s, TTFT p50, TPOT p50 |
|---|---|---|
| 1 | 1.44, 34 ms, 5.19 ms | 1.38, 63 ms, 5.20 ms |
| 8 | 9.02, 24 ms, 6.55 ms | 8.79, 38 ms, 6.57 ms |
| 32 | 22.4, 59 ms, 8.43 ms | 19.8, 120 ms, 9.56 ms |

Decode cost per token is unchanged at low load; saturated throughput is 11% lower. Part of the TTFT gap is cache warmth:
the Serve engine started cold, while the bare-vLLM sweep ran on a warm server (bare vLLM cold TTFT at concurrency 1:
57–65 ms, vs 63 ms through Serve).

**Served task requests** (500 held-out task requests, sent one at a time at simulated time 2026-10-01T04:00Z, 2 h after
the watermark):

| Run | HTTP 200 | Correct vs oracle | First attempt | Verified / abstained / withheld | E2E p50 / p95 |
|---|---|---|---|---|---|
| v1 (`tasks_v1_parser_bug`) | 500 | 0.630 | 0.558 | 210 / 146 / 144 | 776 / 1,544 ms |
| v2, parser fixed (`tasks_v2_parser_fixed`) | 500 | **0.824** | 0.760 | 295 / 171 / 34 | 766 / 1,503 ms |

v1 traces exposed a request-parser bug: categories came out in dictionary order instead of request order, and
"keyboard mouse combo" was split into three categories. Correctness was 0.856 when the parse matched the task and 0.458
when it didn't. After the fix, all 500 parses match. Stage p50s (v2): validate 0.2 ms, retrieval 4.6 ms, Redis features
3.7 ms, solver 0.2 ms, LLM + verification 757 ms. Served correctness isn't directly comparable to the offline 0.748: the
served path builds its own evidence from live features at a different time, and the oracle is the solver on that evidence.

**Freshness and outage probes:** 2 h snapshot age → verified bundle; 7.5 h → `stale_features` (no price or stock
claims made); Redis stopped → HTTP 503 `features_unavailable`.

**Replica failure** (`failover_v2`): after 10 requests, the Generator replica actor was killed (`ray.kill`, no restart).
The Serve controller started a replacement, which reloaded vLLM. The request in flight completed after **22.5 s** with
HTTP 200, and **0 of 60 requests failed**. The first attempt (`failover`) didn't run its drill: it looked up actors
through the dashboard state API, which isn't running. The fix lists named actors through the GCS instead.

Failed attempts kept: `failed_ray_serve_fastapi_pickle` (FastAPI 0.143 app can't be pickled by Ray 2.49; switched to
`__call__` routing) and `failed_ray_serve_protobuf7` (Ray 2.49 needs protobuf < 7; pinned 5.29.5).
Operational note: after `ray stop`, vLLM's `EngineCore` subprocess survived and held 20 GB on GPU 1 until killed; the
runners now kill it explicitly.

## Energy and cost (`energy.json`, `python3 -m cartograph.energy`)
GPU board power sampled every ~0.5 s, integrated over each benchmark's window (CPU, memory, and PSU losses excluded).
Short runs have only 10–20 samples, so treat their values as ±10–15%.

| Config | Mean GPU power | GPU energy per request | GPU Wh per 1k output tokens |
|---|---|---|---|
| bf16, concurrency 1 | 348 W | 66.9 mWh | 0.523 |
| bf16, concurrency 8 | 336 W | 9.9 mWh | 0.077 |
| bf16, concurrency 32 | 344 W | 3.7 mWh | 0.029 |
| FP8, concurrency 8 | 341 W | 6.9 mWh | 0.054 |
| bnb 4-bit, concurrency 8 | 343 W | 17.5 mWh | 0.137 |
| Ray Serve, concurrency 32 | 331 W | 4.2 mWh | 0.033 |
| bf16, max-num-seqs 4 | 346 W | 22.0 mWh | 0.172 |

- **The GPU runs at about its 350 W limit even at concurrency 1**, so energy per request is almost entirely set by
  throughput. Batching to concurrency 32 cuts energy per request about 18×.
- This also bears on the GPU 0 failures: one 3090 already draws about 345 W under this workload, so two together plus
  the CPU approach 1 kW, consistent with the power-delivery hypothesis (not proven).
- **Cost formula:** `cost per request = Wh per request / 1000 × electricity $/kWh`. For illustration only, at an
  *assumed* $0.15/kWh: 1M requests at concurrency 32 ≈ 3.7 kWh ≈ $0.56 of GPU electricity, vs ≈ $10 at concurrency 1.
  Hardware amortization, cooling, and the host's other power draw aren't included.
