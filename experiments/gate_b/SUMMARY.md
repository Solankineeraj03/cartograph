# Gate B: retrieval research summary

All quality numbers are full-catalog retrieval over 482,105 products. Metrics: NDCG@10 with gains E=1, S=0.1, C=0.01, I=0
(unjudged = 0 gain), and Recall@50 over judged E/S products. Dev (2,089 queries) was used for every choice; test (8,956)
is reported once per variant. Per-query outputs, configs, locks, and manifests are in each directory.

## Training budget (identical for every variant)

1,000 steps × 32 queries (one sampled Exact positive each) plus one query-aware hard negative (the query's own judged
I/C product, when it has one), in-batch negatives, temperature 0.05, seed 13, 256/32 document/query tokens.
Known E/S products of the anchor query are masked out of its negatives (45–55 masked entries per run). Only
`split == "train"` judgments are used. b1–b5 trained on CPU (fp32). b6 trained on CPU for the controlled comparison;
a GPU 1 run of b6 is kept separately (`train_b6_raw_cls_gpu1`).

## Objectives and pooling

| ID | Start | Pooling | Objective | Dev NDCG@10 | Test NDCG@10 | Test Recall@50 |
|---|---|---|---|---|---|---|
| — | BM25 (Gate A) | — | — | — | 0.302 | 0.387 |
| — | all-MiniLM-L6-v2, frozen | mean | — | 0.280 | 0.265 | 0.373 |
| b1 | all-MiniLM-L6-v2 | mean | contrastive | **0.321** | **0.308** | **0.415** |
| b2 | all-MiniLM-L6-v2 | mean | graded pairwise (E>S>C>I) | 0.254 | 0.244 | 0.329 |
| b3 | all-MiniLM-L6-v2 | mean | contrastive + graded | 0.311 | 0.299 | 0.403 |
| b4 | all-MiniLM-L6-v2 | mean | contrastive, Matryoshka 384/256/128/64 | 0.310 | 0.299 | 0.408 |
| b5 | MiniLM-L6-H384 (raw) | mean | contrastive | 0.184 | 0.178 | 0.262 |
| b6 | MiniLM-L6-H384 (raw) | CLS | contrastive | 0.150 | 0.144 | 0.215 |

Findings:
- Contrastive fine-tuning with query-aware negatives (b1) raises test NDCG@10 from 0.265 to 0.308 (+16%) and passes BM25
  (0.302). Gate A's fixed fusion of BM25 with the frozen encoder reached 0.322; fusion with b1 hasn't been measured yet.
- **The graded pairwise objective alone hurts (b2 < frozen), and adding it to contrastive also hurts (b3 < b1).**
  This is a negative result for this loss, weight scheme (rank gap / 3), and budget. A likely cause: within-query pairs
  only reorder already-retrieved candidates and give no signal against the rest of the catalog, so full-catalog retrieval
  degrades. It hasn't been measured in the judged-list setting, where it might help.
- Pooling, from the raw backbone at an equal budget: mean beats CLS (0.178 vs 0.144 test). Neither comes close to starting
  from the sentence-pretrained checkpoint; 1,000 steps can't replace sentence-level pretraining.
- Matryoshka training costs native-size quality (0.299 vs 0.308) to make prefixes usable.

## Dimensionality (b1 vs Matryoshka b4), test NDCG@10

| Dims | b1 + train-fitted PCA | b1 truncated (control) | b4 Matryoshka prefix | Raw fp32 storage |
|---|---|---|---|---|
| 384 | **0.308** | — | 0.299 | 706 MiB |
| 256 | **0.302** (98.1%) | 0.296 | 0.289 | 471 MiB |
| 128 | **0.273** | 0.264 | 0.268 | 235 MiB |
| 64 | 0.212 | 0.211 | **0.228** | 118 MiB |

PCA is fit only on vectors of products judged for training queries (`dims_b1_*`). Matryoshka only pays off at 64 dims.
For this encoder, post-hoc PCA beats Matryoshka training at 256 and 128 dims.

## Selection (decided on dev)

**b1 + PCA-256**. Rule: smallest dimension whose dev NDCG@10 is at least 98% of the best native model's.
Dev: 0.3143 / 0.3207 = 98.0% (PCA-128 is 88.8%). Test confirms 98.1% (0.302 vs 0.308) with 33% less raw vector storage.
Qdrant index size and latency for this setting are in `ann_b1_pca256/` (when complete). Hybrid fusion of b1 with BM25,
tuned on dev, is still pending.

## Qdrant for the fine-tuned encoder (2,000 dev / 2,000 test query sample, rule: lowest dev p95 with recall@10 ≥ 0.95)

| Vectors | Raw fp32 | Selected build | Test recall@10 vs exact | p95 latency | NDCG@10 ANN (Δ vs exact, 95% CI) | Qdrant RSS (m32 build) |
|---|---|---|---|---|---|---|
| b1 native 384 | 706 MiB | m16 / efc100 / ef64 | 0.972 | 3.9 ms | 0.312 [−0.009, −0.004] | 2,496 MiB |
| b1 PCA-256 | 471 MiB | m16 / efc100 / ef32 | 0.953 | 3.7 ms | 0.304 [−0.011, −0.006] | 1,499 MiB |

Correction (2026-10-18): an earlier version of this note claimed PCA-256 cuts Qdrant resident memory by about 40%.
That holds only for the m=32 build (1,499 vs 2,496 MiB). For the m=16 build, the one selected, PCA-256 RSS was higher
(1,884 vs 1,706 MiB). RSS sampled right after a build depends on optimizer timing, and on-disk sizes varied just as much
(870–1,913 MiB). Only the raw vector saving is claimed: 471 vs 706 MiB (−33%).

## Hybrid retrieval (BM25 + fine-tuned encoder), selected on dev (`fusion_*`)

Grid: RRF with k ∈ {10, 30, 60, 100} and weighted min-max score fusion with dense weight α ∈ {0.1, …, 0.9}, over the
top-100 of each system. Selection by dev NDCG@10; test reported once.

| Dense vectors | Selected | Dev NDCG@10 | Test NDCG@10 | Δ vs dense alone (95% CI) | Test Recall@50 | Test judged@10 |
|---|---|---|---|---|---|---|
| b1 + PCA-256 | weighted α=0.5 | 0.367 | **0.353** | +0.050 [+0.047, +0.054] | 0.454 | 0.370 |
| b1 native 384 | weighted α=0.5 | 0.369 | 0.355 | +0.047 [+0.043, +0.050] | 0.457 | 0.372 |

Best RRF on dev: k=10 (0.362 / 0.364). Weighted fusion beats RRF here.

**Final retrieval choice: BM25 + b1/PCA-256, weighted α=0.5.** It is 17% above BM25 alone (0.302) and above Gate A's
fusion with the frozen encoder (0.322), at 99.5% of the full-size hybrid's quality.

Correction kept on record: the first fusion runs (`superseded_fusion_*_rrf_k_ignored`) used an `rrf()` that silently
ignored its `k`, so their RRF rows were all k=60. The weighted-fusion selection was unaffected. The function was
fixed, a test now checks that k changes the order, and both runs were repeated.
