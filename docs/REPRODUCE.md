# Reproducing Cartograph

Every experiment directory under `experiments/` has a `manifest.json` with the exact command, code commit and dirty
flag, input and output checksums, environment lock hash, hardware, and duration. This guide lists the steps in order.
Paths assume the layout used on the lab host (`~/cartograph/{repo,data,runs,checkpoints,venvs,opt}`).

## 0. Environments (Linux x86_64, Python 3.11.13 via uv 0.8.22)
```
uv venv ~/cartograph/venvs/<env> --python 3.11.13
VIRTUAL_ENV=~/cartograph/venvs/<env> uv pip sync envs/<env>.lock.txt --index-strategy unsafe-best-match \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu124   # cu128 for training/serving/serve_app
```
| Env | Used for |
|---|---|
| retrieval | ESCI slice, BM25 / encoders, PCA, Qdrant ANN, fusion, governance drill |
| platform | simulator, Ray Data + Iceberg + Feast + Redis drills, bundle tasks |
| training | SFT, merge, DPO, GRPO |
| serving | vLLM evaluation and `vllm bench` |
| serve_app | Ray Serve request path (vLLM + Feast/Redis; protobuf pinned) |

Servers: Redis 7.4.6 (source tarball, SHA-256 `73b94484…`), Qdrant 1.19.0 (musl binary, `9ec66745…`).
Mac-side unit tests need only Python 3.10+ with pandas, pyarrow, scipy: `python3 -m unittest discover -s tests -t .`

## 1. Data and retrieval (Gates A–B)
```
python -m cartograph.esci --raw <esci@7916cdf6> --out data/processed/esci_us_small_v1 --evidence runs/gate_a/esci_slice
python -m cartograph.eval_retrieval --slice <slice> --split test --evidence runs/gate_a/baselines_test   # add --device cpu if needed
bash scripts/gate_b_train_cpu.sh            # six encoder variants + dev/test evaluation (or scripts/gate_b_gpu.sh <gpu> <variants>)
python -m cartograph.dims --slice <slice> --product-vectors <b1>/products.npy --query-vectors <b1>/queries_dev.npy \
  --split dev --dims 256 --save-pca 256 --save-dir data/vectors/b1_pca256 --evidence runs/gate_b/pca256_b1_fit
python -m cartograph.ann --slice <slice> --vectors data/vectors/b1_pca256 --qdrant <qdrant> --work <dir> --evidence runs/gate_b/ann_b1_pca256
python -m cartograph.fusion --slice <slice> --vectors data/vectors/b1_pca256 --evidence runs/gate_b/fusion_b1_pca256
```

## 2. Feature platform (Gate C)
```
python -m cartograph.simulator --slice <slice> --out data/ops_sim_v1
python -m cartograph.gate_c --sim data/ops_sim_v1 --work work/gate_c_v5 --evidence runs/gate_c/drills_v5 --redis-server <redis-server>
```

## 3. Assistant, alignment, serving (Gates D–F), one GPU
```
python -m cartograph.tasks --slice <slice> --sim data/ops_sim_v1 --export work/gate_c_v5/exports/ingest_2 --out data/tasks_v1
GPU=1 bash scripts/gate_d.sh      # solver, base model, SFT pilot + full, SFT eval
GPU=1 bash scripts/gate_e.sh      # merge, GRPO (binary and graded), DPO, evals
GPU=1 bash scripts/gate_f.sh      # Transformers reference, vLLM sweeps, batch limits, prefix cache, quantization
GPU=1 bash scripts/gate_f_serve_v2.sh   # Ray Serve request path: served tasks + replica failover
```
**Run one GPU job at a time.** On the lab host, dual-GPU load has twice triggered Xid 79 on GPU 0.

## 4. Governance and report (Gates G–H)
```
python -m cartograph.gate_g_drill --slice <slice> --sim data/ops_sim_v1 --export work/gate_c_v5/exports/ingest_2 \
  --tasks data/tasks_v1/tasks_test.jsonl --runs runs --artifacts ~/cartograph --evidence runs/gate_g/governance_drill
./scripts/kuberay_kind.sh          # Mac with Docker Desktop + kind, off-VPN; CPU-only KubeRay drill
python3 -m cartograph.report_figures --out docs/figures
```

## Determinism and its limits
Seeds are fixed (simulator 20260901, tasks 101/202, encoders 13, alignment 13, vLLM seed 0, greedy decoding). Results
that depend on GPU kernels (bf16 training, vLLM batching) can differ slightly between runs and hardware. CPU-trained
encoder variants (b1–b6) used fp32. Every dataset and artifact is pinned by SHA-256 in the manifests.
