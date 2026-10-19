# Gate A baseline: BM25 vs untouched MiniLM vs RRF (official US test split)

Status: `measured_local`. Run on gpu-host, CPU only (16 threads), because GPU 0 failed with Xid 79 on the first attempt
(see `../failed_baselines_test_gpu0_xid79`). Code commit `19cd62e`, clean tree. Slice `728e301ef1fb`. Duration 5,154 s.

Reproduce on the host:
```
CUDA_VISIBLE_DEVICES= HF_HOME=~/cartograph/hf ~/cartograph/venvs/retrieval/bin/python -m cartograph.eval_retrieval \
  --slice ~/cartograph/data/processed/esci_us_small_v1 --split test --device cpu --threads 16 --batch-size 128 \
  --latency-queries 500 --evidence <dir> --embedding-cache <dir>/728e301ef1fb.npy
```

## Setup
- 8,956 official test queries; catalog of 482,105 US products (every product judged in the slice).
- Gains E=1, S=0.1, C=0.01, I=0 (the ESCI Task 1 convention). Recall counts E and S as relevant.
  Unjudged results get zero gain and are counted in judged@k rather than treated as irrelevant.
- BM25 k1=1.5, b=0.75, not tuned. MiniLM `all-MiniLM-L6-v2@1110a243`, max 256 tokens, cosine, exact search.
  RRF k=60 over the two top-100 lists, fixed and not tuned. No test data was used to choose anything.

## Results (mean, paired bootstrap 95% CI, 1,000 resamples)

| Setting | Metric | BM25 | MiniLM | RRF |
|---|---|---|---|---|
| Full catalog | NDCG@10 | 0.302 [0.297, 0.308] | 0.265 [0.259, 0.270] | 0.322 [0.315, 0.327] |
| Full catalog | Recall@50 (judged E/S) | 0.387 | 0.373 | 0.432 |
| Full catalog | Judged@10 | 0.335 | 0.288 | 0.349 |
| Judged list | NDCG@10 | 0.699 [0.695, 0.704] | 0.712 [0.708, 0.717] | 0.717 [0.713, 0.721] |

Paired deltas vs BM25, judged list: MiniLM +0.013 [0.010, 0.017], RRF +0.018 [0.016, 0.020]. Per-system deltas
for every metric are in `metrics.json`.

## Reading
- The two settings disagree. MiniLM orders the judged candidates slightly better than BM25, but over the full
  catalog it scores lower. It also returns fewer judged products (judged@10 0.288 vs 0.335), so some of its gap
  may be unjudged products that are actually relevant. Full-catalog NDCG is a lower bound; a manual judgment of
  a sample of unjudged hits is still needed before saying which is better.
- Fusion beats both in every setting and has the best full-catalog Recall@50 (0.432).
- Judged-list Recall@50 is about 1.0 for all systems because the median judged list has 16 products (only 40 of
  8,956 have 50 or more). It carries no information and is reported only for completeness.
- Slices (full-catalog NDCG@10): negation queries are hard for every system (BM25 0.261, MiniLM 0.213). One-to-two-token
  queries hurt MiniLM most (0.210 vs BM25 0.273). The 30 `nlqec` queries score near zero.
- BM25 returned nothing for 50 queries whose tokens are all out of vocabulary (typos such as "amanzon", "berkee").
  Those queries are excluded from judged@10 averages (n=8,906) and score 0 on NDCG.
- Caveat: MiniLM's pretraining data includes Amazon QA text. "Untouched" means not fine-tuned here, not
  "never saw Amazon text".

## Latency (CPU, not comparable to GPU serving)
BM25 full-catalog search: p50 2.1 ms, p95 4.5 ms, p99 5.0 ms (one thread, scipy sparse). MiniLM query encoding at
batch 1: p50 6.6 ms. Exact dense search over 482k × 384 fp32 on CPU: p50 65 ms. Embedding the catalog on CPU took
4,387 s (110 products/s); this is the cost the failed GPU run was meant to avoid.
