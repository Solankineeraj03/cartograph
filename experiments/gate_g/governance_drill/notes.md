# Gate G: governance drill (probes, release gate, rollback, replay)

Status: `measured_local` (CPU). 49 s. Release manifests (`release_*.json`) bind the data slice, Iceberg snapshot IDs
and the feature export Parquet checksums (ingest_2), the PCA-256 basis, encoder b1 weights, the generator adapter
weights, the verifier/solver/assistant/retrieval/features source hashes, every environment lock, and git state.
Gating metrics are read from measured evidence files, not typed in.

## Verifier probes (500 held-out tasks, 3,481 adversarial replies): 100% rejected
| Probe | Rejected |
|---|---|
| Fabricated product ID | 403 / 403 |
| Duplicate product | 403 / 403 |
| False total (−100 cents) | 403 / 403 |
| Out-of-stock substitution | 403 / 403 |
| Over-budget combination | 369 / 369 |
| False abstention on a feasible task | 403 / 403 |
| Bundle offered for an infeasible task | 97 / 97 |
| Format-only `{"decision": "bundle"}` | 500 / 500 |
| Not a JSON object | 500 / 500 |

## Releases (rules: NDCG@10 drop ≤ 2% relative, task correctness drop ≤ 1 point, 0 parity mismatches, all probes rejected)
| Step | Release | Decision |
|---|---|---|
| promote | v1 `2406781e60ea`: b1+PCA-256 (test NDCG 0.302), SFT adapter (wrapped 0.748) | accept (first release) |
| promote | candidate `9f88347d33c2`: b1 truncated to 64 dims (NDCG 0.2105) | **reject**: NDCG relative drop 0.303 > 0.02 |
| promote | v2 `31504bd1c8ea`: GRPO adapter (wrapped 0.754) | accept |
| rollback | v2 → v1 | Gate E found GRPO lowered bundle optimality 0.83 → 0.59. The gate didn't check optimality; that rule should be added |

## Replay after rollback (deterministic path: parse → BM25 over titles per category → as-of features → solver → verifier)
All four saved requests replay with an identical trace digest (release, evidence IDs, decision, verifier, feature snapshot):

| Request | Decision | Trace |
|---|---|---|
| "quiet wireless keyboard and mouse under $100" | bundle: Seenda wireless keyboard 2,699 + Macally wireless mouse 2,199 = 4,898 cents | 11.7 ms total |
| "wireless keyboard and mouse under $5" | infeasible: cheapest in-stock combination is 5,598 cents | 8.3 ms |
| "something nice for my desk" | invalid request: no budget | — |
| "webcam and microphone under $150" | bundle 6,998 cents | 8.3 ms |

Limitation shown by these traces: the deterministic path ranks candidates by BM25 over titles plus the derived category.
For "microphone" it picked a children's karaoke machine, and the "quiet" adjective is never actually verified.
Constraint satisfaction is exact; relevance is only as good as retrieval. The fine-tuned dense encoder, hybrid fusion,
and the LLM explanation are meant to cover this but aren't yet wired into this path.
