# Progress ledger

Statuses: `planned` · `implemented_unrun` · `measured_local` · `measured_cloud` · `blocked_on_hardware`.
A row moves to `measured_*` only when its evidence directory (config, environment lock, manifest, metrics,
per-example outputs, logs, notes) is committed under `experiments/`.

Last updated: 2026-10-18.

## Environment inventory

| Machine | Facts (observed) | Role |
|---|---|---|
| Mac workstation | Apple M2, 8 GB RAM, ~18 GB free disk, Python 3.10.0, no NVIDIA GPU; Docker CLI present (daemon not running), kubectl (no contexts) | Code, unit tests, reports |
| `gpu-host` (ssh gpu-host) | 2× RTX 3090 24 GB, driver 570.207 / CUDA 12.8, PCIe `NODE` topology (no NVLink), Threadripper 3960X (48 threads), 251 GB RAM, 559 GB free disk, shared with other users, no container runtime access | All data, CPU jobs. GPU jobs approved by the owner, one at a time |

**GPU incident (2026-10-14):** during the first Gate A run, GPU 0 logged `Xid 79: GPU has fallen off the bus`, and the
driver set `Node Reboot Required` for both GPUs. It is still unrecovered as of 2026-10-15 01:35 (`nvidia-smi` cannot
open GPU 0; uptime 8 days, so no reboot yet). The same GPU had logged Xid 31 and Xid 43 under other users' processes two
days earlier. An administrator with root needs to reboot the host and check GPU 0. **Second incident (2026-10-15):** after the reboot both GPUs passed a matmul check, but GPU 0 hit Xid 79 again
about a minute into a catalog-encoding job (`experiments/gate_b/failed_gpu_xid79_run2/`), and both GPUs are again flagged for a reboot.
GPU 0 looks faulty in hardware (power, PCIe, or seating). **No GPU work runs until an administrator inspects GPU 0.**
Evidence: `experiments/gate_a/failed_baselines_test_gpu0_xid79/`.

**Connectivity:** the host is reachable only from the university network or VPN. SSH dropped twice while jobs ran;
remote jobs are started with `setsid nohup` and keep running.

## Environments (hashed locks in `envs/`)

| Env | Key pins | Status |
|---|---|---|
| retrieval | torch 2.6.0+cu124, sentence-transformers 3.4.1, transformers 4.48.3, qdrant-client 1.19.0 | built; CUDA smoke test passed before the incident |
| platform | ray[data] 2.58.0, pyiceberg 0.12.0, feast[redis] 0.66.0, redis-py 7.4.1 | built, in use |
| training | torch 2.8.0 (cu128), transformers 4.57.1, trl 0.24.0, peft 0.17.1, datasets 4.2.0 | built; not yet exercised |
| serving | vllm 0.11.0, ray[serve] 2.49.2 | built; not yet exercised |
| servers | Redis 7.4.6 (built from source), Qdrant 1.19.0 (release binary), user space under `~/cartograph/opt` | in use |

Models pinned: Qwen2.5-1.5B-Instruct `989aa798`, -AWQ `3ecffa0c`, -GPTQ-Int4 `4f5a0e31`, Qwen2.5-0.5B-Instruct `7ae55760`,
all-MiniLM-L6-v2 `1110a243`, MiniLM-L6-H384-uncased `3276f0fa`.

## Gates

| Gate | Item | Status | Evidence / notes |
|---|---|---|---|
| A | Inventory, Git, env locks | measured_local | Above; repo history from `299c547` |
| A | ESCI source, license, checksums | measured_local | Commit `7916cdf6`, Apache-2.0, SHA-256 match LFS pointers |
| A | Slice, split rules, duplicate/overlap audit | measured_local | `experiments/gate_a/esci_slice/` (slice `728e301ef1fb`): 18,777 train / 2,089 dev / 8,956 test queries; 22 train groups overlapping test excluded; 0 duplicate pairs; 221 cross-query label conflicts; 74% of test judgments on products unseen in train |
| A | BM25 vs untouched MiniLM vs RRF | measured_local (CPU) | `experiments/gate_a/baselines_test_cpu/`. Full-catalog NDCG@10 0.302 / 0.265 / 0.322; judged-list 0.699 / 0.712 / 0.717 |
| B | Train-fitted PCA vs truncation (frozen encoder) | measured_local (CPU) | `experiments/gate_b/dims_frozen_test/`: no reduction keeps 98% NDCG; PCA ≈ truncation |
| B | Qdrant HNSW vs exact | measured_local (CPU) | `experiments/gate_b/ann_frozen/`: selected m32/efc200/ef32, test recall@10 0.962, p95 3.8 ms, NDCG Δ −0.005 |
| B | Contrastive / graded / combined (b1–b3) | measured_local (CPU) | `experiments/gate_b/{train,eval}_b[123]_*`. Test NDCG@10 full catalog: contrastive 0.308, graded 0.244, combined 0.299 (frozen 0.265, BM25 0.302) |
| B | Matryoshka (b4), mean vs CLS (b5/b6), compression, selection | measured_local (CPU) | `experiments/gate_b/SUMMARY.md`. Selected b1 + train-fitted PCA-256: 98.1% of native test NDCG, ~40% less Qdrant RSS. Graded loss is a negative result; mean > CLS; Matryoshka only wins at 64 dims |
| B | Hybrid fusion tuned on dev | measured_local | `experiments/gate_b/fusion_b1_pca256`: weighted α=0.5, test NDCG@10 0.353 (+0.050 [+0.047, +0.054] over dense). The first run's RRF k sweep was buggy, superseded, and kept |
| C | Simulator, Ray Data, Iceberg, Feast, Redis drills | measured_local (CPU) | `experiments/gate_c/drills_v5/notes.md`: all checks pass (historical vs reference 5,001/5,001; online/offline parity 0 mismatches; replay idempotent; worker-kill retry; late data never rewrites history; fail-closed on Redis outage). Runs 1–4 failed, kept with notes |
| D | Bundle tasks, solver, verifier, scoring | measured_local | `experiments/gate_d/`: 3,000 train / 500 test tasks, 0 shared products; solver 500/500 |
| D | Base model vs SFT (LoRA) on GPU 1 | measured_local | `experiments/gate_d/SUMMARY.md`: correct, raw/wrapped: base 0.000/0.190 (refuses everything), SFT 0.688/0.748; solver 1.000 |
| D | Served request path (Ray Serve + live Feast/Redis features + in-process vLLM, verify/retry, traces) | measured_local | `experiments/gate_f/ray_serve/`: 500/500 HTTP 200; fresh / stale / Redis-down probes behave as designed; parser bug found and fixed (see v2) |
| E | GRPO (RLVR, verifier reward), DPO (verifier pairs, synthetic) | measured_local | `experiments/gate_e/SUMMARY.md`: neither significantly beats SFT held out (GRPO +0.012 [−0.022, +0.044]; DPO −0.002). GRPO cut over-budget errors but dropped optimality 0.83 → 0.59 (binary-reward hacking). 88% of GRPO groups all-equal |
| E | Reward model | planned | Code ready (`align.py rm`); it is only meaningful on human labels |
| E | Human preference labels → RLHF | blocked on owner labels and hardware | `cartograph/label.py`, `docs/PREFERENCE_RUBRIC.md`; the GRPO-on-RM path refuses non-human reward models |
| F | Serving: Transformers vs vLLM vs Ray Serve, concurrency/rate sweeps, batch limits, KV capacity, prefix cache, FP8/bnb quantization with quality | measured_local | `experiments/gate_f/SUMMARY.md` |
| G | TP=1 vs TP=2 | blocked_on_hardware | Needs two healthy GPUs. GPU 0 fell off the bus twice under dual-GPU load; excluded pending admin inspection or a power cap |
| G | KubeRay | implemented_unrun | `deploy/kuberay/`, `scripts/kuberay_kind.sh`. The Mac Docker Desktop VM can't pull images while the VPN is up (Docker settings were tried and restored); run it off-VPN |
| G | Cloud run (TP=1 vs TP=2, GPU KubeRay) | planned | No account yet; exact plan in `docs/CLOUD_PLAN.md` |
| G | Release gate, rollback, replay | measured_local | `experiments/gate_g/governance_drill`: 3,481/3,481 probes rejected; 64-dim release rejected; rollback + identical replay |
| H | Report, model/data cards, reproducibility guide, figures | measured_local | `docs/REPORT.md`, `docs/MODEL_CARD.md`, `docs/DATA_CARD.md`, `docs/REPRODUCE.md`, `docs/figures/` |

## Session log

- 2026-10-14: Inventory, Git, environments, ESCI download/audit. The GPU 0 Xid 79 incident stopped GPU work; the
  quality baseline was rerun on CPU (86 min). Gate C run 1 failed on nanosecond timestamps (fixed in the simulator).
- 2026-10-15: Gate C run 2 stalled for 5.7 h on a single-block Ray shuffle (fixed with per-bucket blocks, no
  shuffle). Gate B frozen-encoder dimensionality and Qdrant ANN measured. Training/serving envs built.
  Six-variant CPU encoder training launched. SSH to the host dropped again at about 02:00.

- 2026-10-17: Host rebooted. Gate C run 4 failed (Feast file store drops TTL-expired entity rows); run 5 passed. Gate B finished on CPU (all six variants) and selection made. Gate D measured on GPU 1: tasks, solver, base model, SFT. The SFT truncation bug was caught by the memory pilot and guarded. Gate E launched on GPU 1.
- 2026-10-18: Gate F measured (vLLM sweeps, prefix cache, FP8/bnb with quality, Ray Serve). Gate E round 2 (graded GRPO). Gate B fusion (RRF k bug fixed and rerun). Gate G governance drill. Served path: FastAPI-pickling and protobuf-7 incompatibilities fixed; a parser bug (category order, combo phrases) found from served traces and fixed. KubeRay blocked by Docker Desktop networking on VPN. Report, cards, reproducibility guide.
