# Gate C: feature platform drills (run 5, all checks passed)

Status: `measured_local` (CPU, gpu-host). Commit `b00009c`, clean tree, 1,258 s. Runs 1–4 failed and are kept
next to this directory (`failed_drills_v1..v4`), each with notes on its cause and fix.

All operational data is synthetic (`ops-sim-v1`, seed 20260901): 482,105 products, 30 days, 5.05M events in two
delivery batches (cutoff 2026-09-21), with planted late deliveries, duplicate deliveries, and invalid rows.

| Check | Result |
|---|---|
| Ray Data on two raylets (one host, 8 CPUs), one worker killed with `os._exit` on bucket 7 | Retried; committed event IDs equal a pandas reference run |
| Validation and dedup, batch 1 | 3,606,157 raw → 3,584,780 committed; 17,887 duplicate deliveries removed; 3,490 invalid prices quarantined; 0 duplicate IDs committed |
| Replay batch 1 | 0 events appended; events snapshot unchanged |
| Batch 2 (late deliveries) | 1,440,410 appended; 1 duplicate whose first copy came in batch 1 skipped |
| Iceberg tags `ingest_1`, `ingest_2` → immutable Parquet export with checksums | `export_manifest_ingest_*.json` |
| Schema evolution (column added after tagging) | ingest_1 snapshot still readable with its original schema (1,117,488 price rows) |
| Feast historical vs independent reference (`latest_available` over raw events, same TTL) | 5,001 samples: price, stock, popularity 5,001 / 5,001 match |
| Naive event-time join (ignores availability) | Would disagree on 33 price and 20 stock samples, i.e. leak late data |
| Late data never rewrites history | 2,981 / 2,981 as-of values before the boundary identical across ingest_1 and ingest_2 exports |
| Materialize to Redis twice | 482,104 keys, byte-identical contents (49 s, then 39 s) |
| Online vs offline at the materialization time | 2,000 products × 8 features, 0 mismatches |
| Missing product / stale watermark | `missing`; `stale` at 7 h snapshot age (limit 6 h) |
| Batched online fetch, 50 products | p50 4.6 ms, p95 6.0 ms, p99 6.5 ms (n = 500) |
| Redis outage | `FeaturesUnavailable` raised; no values served |

Design findings from the failed runs:
- Feast point-in-time joins use event time only. Availability-correct history comes from the export: a row is emitted
  at `available_at`, and only if it is newer than every event already known.
- Feast's file offline store drops the entire entity row when any requested view has no row inside its TTL, instead of
  returning nulls. Fixes: query views separately, and use a long TTL for change-only features (price, stock), with
  freshness enforced by the snapshot watermark.
- Rows reported missing by view (popularity 168, price 16, stock 25 of 5,001) are cutoffs before a product's first
  available row or past the 2-day popularity TTL. The reference returns null for exactly those, and they count as matches.
- "Two raylets on one host" exercises Ray's scheduling and retry paths, but it isn't a multi-machine cluster.
