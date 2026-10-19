# R7: Qdrant HNSW vs exact search (frozen MiniLM vectors, identical vectors in both)

Status: `measured_local` (CPU). Qdrant 1.19.0 binary on gpu-host, qdrant-client 1.19.0, dot product on
unit vectors (equivalent to cosine). 482,105 vectors × 384 fp32 (706 MiB raw). A seeded sample of 2,000 dev and
2,000 test queries. Settings were selected on dev only, with the rule: lowest dev p95 latency with dev ANN recall@10 ≥ 0.95.

ANN recall here is agreement with exact top-k search, not relevance.

| Build | Build time | Collection on disk | Qdrant RSS | hnsw_ef | Test recall@10 vs exact | Test recall@100 | p95 latency |
|---|---|---|---|---|---|---|---|
| m=16, efc=100 | 34 s | 1,624 MiB | 1,895 MiB | 64 | 0.961 | 0.947 | 3.9 ms |
| m=16, efc=100 | | | | 256 | 0.989 | 0.987 | 5.0 ms |
| m=32, efc=200 | 53 s | 1,500 MiB | 2,470 MiB | 32 (**selected**) | 0.962 | 0.942 | 3.8 ms |
| m=32, efc=200 | | | | 128 | 0.990 | 0.987 | 4.9 ms |
| m=32, efc=200 | | | | 256 | 0.995 | 0.995 | 6.0 ms |

All builds and ef values are in `metrics.json`. Latency goes through the Python client over localhost HTTP,
one request at a time, so client overhead dominates. Exact search over the same vectors costs about 65 ms per query on CPU (Gate A).

Relevance cost of the selected ANN setting on the 2,000 test queries (paired bootstrap):
NDCG@10 0.2724 → 0.2671, Δ −0.0053 [−0.0076, −0.0029]; Recall@50 Δ −0.0103 [−0.0129, −0.0082].
The loss is small but statistically clear. ef=128 roughly halves the recall gap for about 1 ms more at p95.
On-disk size is about 2.1× the raw vectors (graph, payload storage, WAL); RSS is 2.7–3.5×. The m=16 collection
being larger on disk than m=32 is as measured; it probably reflects segment and WAL state at snapshot time and wasn't investigated.
