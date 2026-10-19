# R5 (partial): dimensionality on frozen MiniLM vectors, official test split

Status: `measured_local` (CPU). This covers native vs train-fitted PCA vs naive truncation on the untouched
encoder only. The Matryoshka-trained comparison needs the fine-tuned encoders and is pending GPU access.

- PCA is fit on 319,972 vectors of products judged for train-split queries only, never on test queries.
- 8,956 test queries, exact full-catalog search. Gains E=1, S=0.1, C=0.01, I=0; unjudged results get zero gain.

| Variant | NDCG@10 | Δ vs native (paired) | Recall@50 | Raw fp32 storage | Exact search ms/query (CPU) |
|---|---|---|---|---|---|
| native 384 | 0.265 | — | 0.373 | 706 MiB | 1.07 |
| truncate 256 | 0.254 | −0.012 | 0.355 | 471 MiB | 0.86 |
| PCA 256 | 0.252 | −0.013 | 0.358 | 471 MiB | 0.87 |
| truncate 128 | 0.224 | −0.041 | 0.306 | 235 MiB | 0.67 |
| PCA 128 | 0.224 | −0.041 | 0.320 | 235 MiB | 0.66 |
| truncate 64 | 0.175 | −0.090 | 0.230 | 118 MiB | 0.60 |
| PCA 64 | 0.172 | −0.093 | 0.252 | 118 MiB | 0.60 |

Search latency here is a batched (256 queries) CPU matmul divided by query count, so it isn't a serving latency.

Findings:
- No variant reaches the 98% NDCG-retention hypothesis. Even 256 dims keeps only about 95.6% (truncation) or 95.0% (PCA).
- PCA doesn't beat truncation on NDCG@10 at any size. It does keep more Recall@50 at 128 and 64 dims (+0.015, +0.021).
  Most of the trained embedding's ranking signal isn't concentrated in its top principal directions.
- This is a negative result for post-hoc compression of this encoder. Matryoshka training is the remaining
  candidate (planned; needs GPU).
