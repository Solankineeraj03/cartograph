# Cartograph

A grounded product-discovery assistant with an evidence-backed ML platform behind it. Ask for "a quiet wireless keyboard
and mouse bundle under $100, only available products". Cartograph retrieves candidates (BM25 + a fine-tuned encoder),
reads point-in-time price and stock features, picks a bundle with a fine-tuned Qwen2.5-1.5B model, and **verifies
every selection** (real product IDs, in stock, one per category, exact total, within budget). If no valid bundle exists,
it says why. Unverifiable answers are never shown.

**Start with [the technical report](docs/REPORT.md).** Every number in it comes from a committed experiment directory
under `experiments/`, with its command, code commit, input checksums, environment lock, and per-example outputs.

## Headline results (all measured; synthetic operational data where noted)

| Area | Result | Evidence |
|---|---|---|
| Retrieval (ESCI US test, 8,956 queries) | Hybrid BM25 + fine-tuned encoder (PCA-256): NDCG@10 **0.353** vs BM25 0.302, frozen encoder 0.265 | `experiments/gate_b/` |
| Compression | Train-fitted PCA-256 keeps 98.1% of NDCG with 33% less vector storage; Matryoshka wins only at 64 dims | `experiments/gate_b/SUMMARY.md` |
| Feature platform | Ray Data → Iceberg → Feast → Redis: 5,001/5,001 point-in-time matches, 0 online/offline mismatches | `experiments/gate_c/drills_v5` |
| Assistant (500 held-out synthetic bundle tasks) | Base model 0.00 → SFT **0.69** raw / **0.75** with verify-and-retry; exact solver 1.00 | `experiments/gate_d/` |
| Alignment | GRPO on verifier rewards: no significant gain; binary reward was gamed (optimality 0.83 → 0.59), graded reward fixed it. DPO: no effect at lr 5e-6, **−6 points** when trained harder (overfit). Not RLHF: human labels pending | `experiments/gate_e/` |
| Serving (one RTX 3090) | vLLM 2,869 vs Transformers 54 output tok/s; prefix cache 3.6× lower TTFT; FP8 weights −42% memory, +23–38% decode speed, −1.4 points | `experiments/gate_f/` |
| Governance | 3,481/3,481 adversarial answers rejected; degraded release blocked; rollback + identical replay | `experiments/gate_g/` |

**Not done:** two-GPU tensor parallelism and the cloud run (`docs/CLOUD_PLAN.md`); KubeRay on Kubernetes (ready in
`deploy/kuberay`, not yet run); human-feedback RL (needs owner preference labels). See [docs/LEDGER.md](docs/LEDGER.md)
for every item's status.

## Website (GitHub Pages, served from `docs/`)
- `docs/index.html`: project page (architecture, results, limits, links to evidence).
- `docs/demo.html`: live demo. Start the backend on the GPU host through an SSH tunnel:
  `ssh -L 8012:127.0.0.1:8012 <gpu-host> 'bash ~/cartograph/repo/scripts/demo_up.sh'`, then open the page and press Connect.
  Ctrl-C or closing SSH stops everything. With no backend, the page replays real recorded responses (`docs/site/recorded.json`,
  regenerate with `python3 -m cartograph.record_demo`). Links: `demo.html#example=N` opens recorded example N; `demo.html#ask` sends the current request.
- Set the repository URL once in `docs/site/config.js`.

## Repository map
- `cartograph/`: all code (retrieval, metrics, verifier, solver, simulator, pipeline, Feast, training, serving, governance).
- `experiments/`: evidence per gate, including failed runs and their notes.
- `docs/`: [report](docs/REPORT.md), [ledger](docs/LEDGER.md), [reproduce](docs/REPRODUCE.md),
  [model card](docs/MODEL_CARD.md), [data card](docs/DATA_CARD.md), [cloud plan](docs/CLOUD_PLAN.md),
  [preference rubric](docs/PREFERENCE_RUBRIC.md), and the original [design](docs/PROJECT.md),
  [experiments protocol](docs/EXPERIMENTS.md), and [roadmap](docs/ROADMAP.md).
- `envs/`: hashed dependency locks. `scripts/`: gate runners. `deploy/kuberay/`: image and RayService manifests.

## Quick checks on any machine
```
python3 -m unittest discover -s tests -t .      # needs pandas, pyarrow, scipy
python3 -m cartograph.demo                      # original dependency-free CPU fixture
python3 -m cartograph.report_figures            # regenerate figures from committed evidence
```
Full reproduction (GPU host): [docs/REPRODUCE.md](docs/REPRODUCE.md).
