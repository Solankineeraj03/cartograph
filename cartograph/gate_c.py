"""Gate C drills: Ray Data -> Iceberg -> Parquet export -> Feast historical -> Redis online.

Each check writes pass/fail counts to metrics.json and examples to per_example.jsonl:
 1. Multi-worker Ray Data ingest on a two-raylet Ray cluster (one physical host), with one worker
    killed mid-task (os._exit) and retried; output event_ids must equal a pandas reference.
 2. Replaying batch 1 appends nothing (idempotent).
 3. Batch 2 (late deliveries) is ingested and tagged; the as-of values at cutoffs before the batch
    boundary must be identical in the ingest_1 and ingest_2 exports (late data never rewrites history).
 4. Schema evolution: add a column after ingest_2; the tagged snapshots still read with old schema.
 5. Feast historical features vs an independent reference built from raw events with
    ``cartograph.features.latest_available``; plus how often a naive event-time join would leak.
 6. Materialize to Redis twice: identical Redis contents; online values equal historical at the same time.
 7. Missing product, stale watermark, batched fetch latency, and Redis outage (fail closed).
"""

import argparse
import hashlib
import json
import random
import subprocess
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import pipeline
from .features import latest_available
from .feast_repo import FEATURES, TTL, objects, write_config
from .online import FeatureClient, FeaturesUnavailable
from .provenance import Run


def iso(ts) -> str:
    return pd.Timestamp(ts).isoformat()


def reference_value(events_by_product: dict, product_id: str, kind: str, cutoff: pd.Timestamp):
    rows = [{"product_id": product_id, "event_time": iso(e), "available_at": iso(a), "payload": p}
            for e, a, p in events_by_product.get((product_id, kind), [])]
    row = latest_available(rows, product_id, iso(cutoff))
    if row is None or cutoff - pd.Timestamp(row["available_at"]) > TTL[kind]:
        # Feast TTL is relative to feature_ts (= available_at of the winning event).
        return None
    return json.loads(row["payload"])


def naive_value(events_by_product: dict, product_id: str, kind: str, cutoff: pd.Timestamp):
    rows = [(e, p) for e, a, p in events_by_product.get((product_id, kind), []) if e <= cutoff]
    return json.loads(max(rows)[1]) if rows else None


def reference_popularity(interactions: pd.DataFrame, product_id: str, cutoff: pd.Timestamp, start, days):
    grace = pipeline.GRACE
    day_ends = [start + pd.Timedelta(days=d + 1) for d in range(days)]
    eligible = [end for end in day_ends if end + grace <= cutoff]
    if not eligible or cutoff - (eligible[-1] + grace) > TTL["popularity"]:
        return None
    end = eligible[-1]
    product = interactions[interactions.product_id == product_id]
    if not (product.available_at <= end + grace).any():
        return None
    window = product[(product.event_time > end - pd.Timedelta(days=7)) & (product.event_time <= end)
                     & (product.available_at <= end + grace)]
    return int(window.views.sum())


def redis_digest(port: int) -> tuple[str, int]:
    import redis
    client = redis.Redis(host="127.0.0.1", port=port)
    digest = hashlib.sha256()
    keys = sorted(k for k in client.scan_iter(count=10000) if not k.startswith(b"cartograph:"))
    for key in keys:
        digest.update(key)
        for field, value in sorted(client.hgetall(key).items()):
            digest.update(field + b"=" + value)
    return digest.hexdigest(), len(keys)


def historical_by_view(feature_store, entity_df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict]:
    """Query each feature view separately, aligned by row_id. Feast's file offline store drops the whole entity row
    when any requested view has no row within its TTL, so a single combined query would hide valid values."""
    frames, missing = [], {}
    for view in sorted({f.split(":")[0] for f in features}):
        wanted = [f for f in features if f.startswith(view + ":")]
        out = feature_store.get_historical_features(entity_df=entity_df, features=wanted).to_df()
        out = out.drop_duplicates("row_id").set_index("row_id")
        missing[view] = int(len(entity_df) - len(out))
        frames.append(out.reindex(entity_df.row_id)[[f.split(":")[1] for f in wanted]])
    return pd.concat(frames, axis=1), missing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", type=Path, required=True, help="simulator output directory")
    parser.add_argument("--work", type=Path, required=True, help="warehouse/feature repo working directory")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--redis-server", type=Path, required=True)
    parser.add_argument("--redis-port", type=int, default=6390)
    parser.add_argument("--cpus-per-node", type=int, default=4)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import ray
    from ray.cluster_utils import Cluster

    sim_manifest = json.loads((args.sim / "simulation_manifest.json").read_text())
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["simulator"] = {k: sim_manifest[k] for k in ("simulator_version", "seed", "config")}
    with Run(args.evidence, "gate_c_feature_platform", config, seed=args.seed,
             inputs={"simulation_manifest": args.sim / "simulation_manifest.json"},
             lock=Path("envs/platform.lock.txt")) as run:
        rng = random.Random(args.seed)
        metrics, examples = {}, []
        warehouse = args.work / "warehouse"
        if warehouse.exists():
            raise FileExistsError(f"{warehouse} exists; use a fresh --work directory")
        landing = args.sim / "landing"

        # 1. Two raylets on this host; kill one worker once while it processes bucket 7.
        cluster = Cluster(initialize_head=True, head_node_args={"num_cpus": args.cpus_per_node})
        cluster.add_node(num_cpus=args.cpus_per_node)
        marker = args.work / "fail_marker.json"
        runtime_env = {"env_vars": {pipeline.FAIL_ENV: "7", pipeline.FAIL_MARKER_ENV: str(marker)}}
        ray.init(address=cluster.address, runtime_env=runtime_env, include_dashboard=False)
        nodes = [n for n in ray.nodes() if n["Alive"]]
        metrics["ray"] = {"version": ray.__version__, "alive_nodes": len(nodes),
                          "cpus": ray.cluster_resources().get("CPU"),
                          "note": "Two raylets on one physical host; not a multi-machine cluster."}
        start = time.perf_counter()
        report_1 = pipeline.ingest(landing, "batch_1", warehouse, sim_manifest, tag="ingest_1")
        metrics["ingest_1"] = {**report_1, "seconds": time.perf_counter() - start}
        metrics["worker_failure"] = {"injected": marker.exists(),
                                     "marker": json.loads(marker.read_text()) if marker.exists() else None}

        raw_1 = pd.concat([pd.read_parquet(p) for p in sorted((landing / "batch_1").glob("*.parquet"))])
        reference = pd.concat([pipeline.validate_bucket(g.drop(columns=[]))
                               for _, g in raw_1.groupby("bucket")])  # marker exists, so no crash here
        store = pipeline.catalog(warehouse)
        committed = store.load_table("ops.events").scan().to_pandas()
        quarantined = store.load_table("ops.events_quarantine").scan().to_pandas()
        ref_good = set(reference.loc[reference.reason == "", "event_id"])
        metrics["ingest_1"]["raw_rows"] = int(len(raw_1))
        metrics["ingest_1"]["planted_duplicates"] = int((raw_1.planted_fault == "duplicate_delivery").sum())
        metrics["ingest_1"]["planted_invalid"] = int((raw_1.planted_fault == "invalid_negative_price").sum())
        metrics["ingest_1"]["matches_pandas_reference"] = set(committed.event_id) == ref_good
        metrics["ingest_1"]["duplicate_event_ids_committed"] = int(committed.event_id.duplicated().sum())
        metrics["ingest_1"]["quarantine_reasons"] = quarantined.reason.value_counts().to_dict()

        # 2. Replay batch 1.
        events_before = store.load_table("ops.events").current_snapshot().snapshot_id
        replay = pipeline.ingest(landing, "batch_1", warehouse, sim_manifest, tag=None)
        events_after = store.load_table("ops.events").current_snapshot().snapshot_id
        metrics["replay_batch_1"] = {"appended_events": replay["appended_events"],
                                     "events_snapshot_unchanged": events_before == events_after}

        # 3. Late deliveries.
        start = time.perf_counter()
        report_2 = pipeline.ingest(landing, "batch_2", warehouse, sim_manifest, tag="ingest_2")
        metrics["ingest_2"] = {**report_2, "seconds": time.perf_counter() - start}
        ray.shutdown()
        cluster.shutdown()

        exports = {tag: pipeline.export(warehouse, tag, args.work / "exports" / tag)
                   for tag in ("ingest_1", "ingest_2")}
        metrics["exports"] = exports

        # 4. Schema evolution after the tags.
        from pyiceberg.types import StringType
        price_table = store.load_table("features.price_state")
        price_table.update_schema().add_column("currency", StringType(), doc="added in drill").commit()
        old = price_table.scan(snapshot_id=price_table.snapshot_by_name("ingest_1").snapshot_id).to_arrow()
        metrics["schema_evolution"] = {"current_columns": [f.name for f in price_table.schema().fields],
                                       "ingest_1_rows_readable": old.num_rows,
                                       "ingest_1_currency_all_null": old.column("currency").null_count == old.num_rows
                                       if "currency" in old.column_names else "column absent in old snapshot read"}

        # Reference data from the tagged ingest_2 events snapshot.
        events_table = store.load_table("ops.events")
        events = events_table.scan(snapshot_id=events_table.snapshot_by_name("ingest_2").snapshot_id).to_pandas()
        events["event_time"] = pd.to_datetime(events.event_time, utc=True)
        events["available_at"] = pd.to_datetime(events.available_at, utc=True)
        by_product = {}
        for row in events[events.event_type != "interaction"].itertuples():
            by_product.setdefault((row.product_id, row.event_type), []).append(
                (row.event_time, row.available_at, row.payload))
        interactions = events[events.event_type == "interaction"].copy()
        payload = interactions.payload.map(json.loads)
        interactions["views"] = payload.map(lambda p: p["views"])

        sim_start = pd.Timestamp(sim_manifest["config"]["start"])
        sim_end = sim_start + pd.Timedelta(days=sim_manifest["config"]["days"])
        cutoff_boundary = pd.Timestamp(sim_manifest["config"]["batch_cutoff"])
        products = sorted(events.product_id.unique())
        late_products = sorted(events.loc[events.available_at - events.event_time > pd.Timedelta(hours=6),
                                          "product_id"].unique())
        sample = []
        for i in range(args.samples):
            # Half the samples target products with late events, near their late event times.
            product = rng.choice(late_products if i % 2 else products)
            cutoff = sim_start + pd.Timedelta(seconds=rng.uniform(0, (sim_end - sim_start).total_seconds()
                                                                  + 4 * 86400))
            sample.append((product, cutoff.floor("s")))
        missing_product = sim_manifest["product_without_events"]
        sample.append((missing_product, sim_end))

        # 5. Feast historical features on the ingest_2 export.
        from feast import FeatureStore
        repo = write_config(args.work / "feature_repo", args.redis_port)
        redis_proc = subprocess.Popen([str(args.redis_server), "--port", str(args.redis_port), "--bind", "127.0.0.1",
                                       "--save", "", "--appendonly", "no"], stdout=subprocess.DEVNULL)
        time.sleep(1.0)
        try:
            feature_store = FeatureStore(repo_path=str(repo))
            feature_store.apply(objects(args.work / "exports" / "ingest_2"))
            # row_id is carried through Feast so results are matched by ID, not by timestamp equality.
            entity_df = pd.DataFrame({"row_id": range(len(sample)), "product_id": [p for p, _ in sample],
                                      "event_timestamp": [c for _, c in sample]})
            start = time.perf_counter()
            historical, missing_by_view = historical_by_view(feature_store, entity_df, FEATURES)
            metrics["historical_seconds"] = time.perf_counter() - start
            metrics["historical_rows_missing_by_view"] = missing_by_view
            checks = {"price": [0, 0], "stock": [0, 0], "popularity": [0, 0]}
            naive_leaks = {"price": 0, "stock": 0}
            for row_id, (product, cutoff) in enumerate(sample):
                row = historical.loc[row_id]
                expected_price = reference_value(by_product, product, "price", cutoff)
                expected_stock = reference_value(by_product, product, "stock", cutoff)
                expected_views = reference_popularity(interactions, product, cutoff, sim_start,
                                                      sim_manifest["config"]["days"])
                got_price = None if pd.isna(row["price_cents"]) else int(row["price_cents"])
                got_stock = None if pd.isna(row["in_stock"]) else bool(row["in_stock"])
                got_views = None if pd.isna(row["views_7d"]) else int(row["views_7d"])
                results = {"price": got_price == (expected_price or {}).get("price_cents"),
                           "stock": got_stock == (expected_stock or {}).get("in_stock"),
                           "popularity": got_views == expected_views}
                for name, ok in results.items():
                    checks[name][0 if ok else 1] += 1
                for kind in ("price", "stock"):
                    naive = naive_value(by_product, product, kind, cutoff)
                    if naive != reference_value(by_product, product, kind, cutoff) and \
                            reference_value(by_product, product, kind, cutoff) is not None:
                        naive_leaks[kind] += 1
                if not all(results.values()) or len(examples) < 20:
                    examples.append({"check": "historical_vs_reference", "product_id": product,
                                     "cutoff": iso(cutoff), "feast": [got_price, got_stock, got_views],
                                     "reference": [expected_price, expected_stock, expected_views],
                                     "ok": all(results.values())})
            metrics["historical_vs_reference"] = {k: {"match": v[0], "mismatch": v[1]} for k, v in checks.items()}
            metrics["naive_event_time_join_disagreements"] = naive_leaks
            metrics["samples"] = len(sample)

            # 3b. Late data never rewrites history: compare as-of values from both exports before the boundary.
            early = [(p, c) for p, c in sample if c <= cutoff_boundary]
            frames = {}
            for tag in ("ingest_1", "ingest_2"):
                feature_store.apply(objects(args.work / "exports" / tag))
                frames[tag] = historical_by_view(feature_store, pd.DataFrame(
                    {"row_id": range(len(early)), "product_id": [p for p, _ in early],
                     "event_timestamp": [c for _, c in early]}),
                    ["price:price_cents", "stock:in_stock"])[0].reset_index(drop=True)
            a, b = frames["ingest_1"], frames["ingest_2"]
            same = ((a.price_cents.fillna(-1) == b.price_cents.fillna(-1)) &
                    (a.in_stock.astype("object").fillna("na") == b.in_stock.astype("object").fillna("na")))
            metrics["history_stable_across_late_ingest"] = {"cutoffs_before_boundary": len(early),
                                                            "identical": int(same.sum()),
                                                            "different": int((~same).sum())}

            # 6. Materialize twice, digest Redis, compare online vs historical at the materialization end.
            feature_store.apply(objects(args.work / "exports" / "ingest_2"))
            end = sim_end + pd.Timedelta(hours=2)
            timings = []
            for _ in range(2):
                start = time.perf_counter()
                feature_store.materialize(start_date=sim_start.to_pydatetime() - timedelta(days=1),
                                          end_date=end.to_pydatetime())
                timings.append(time.perf_counter() - start)
                if _ == 0:
                    first = redis_digest(args.redis_port)
            second = redis_digest(args.redis_port)
            metrics["materialization"] = {"seconds": timings, "keys": second[1],
                                          "repeat_identical": first == second}
            client = FeatureClient(feature_store, args.redis_port)
            client.write_watermark(end.to_pydatetime(), "ingest_2",
                                   {k: v["snapshot_id"] for k, v in exports["ingest_2"]["tables"].items()})
            parity_products = rng.sample(products, min(2000, len(products)))
            online = feature_store.get_online_features(
                features=FEATURES, entity_rows=[{"product_id": p} for p in parity_products]).to_dict()
            offline, parity_missing = historical_by_view(feature_store, pd.DataFrame(
                {"row_id": range(len(parity_products)), "product_id": parity_products,
                 "event_timestamp": [end] * len(parity_products)}), FEATURES)
            metrics["parity_offline_rows_missing_by_view"] = parity_missing
            mismatches = []
            for i, product in enumerate(parity_products):
                for feature in [f.split(":")[1] for f in FEATURES]:
                    a_value, b_value = online[feature][i], offline.loc[i, feature]
                    b_value = None if pd.isna(b_value) else b_value
                    if b_value is not None and isinstance(a_value, (int, float, bool)):
                        equal = type(a_value)(b_value) == a_value
                    else:
                        equal = a_value == b_value
                    if not equal:
                        mismatches.append({"product_id": product, "feature": feature,
                                           "online": repr(a_value), "offline": repr(b_value)})
            metrics["online_offline_parity"] = {"products": len(parity_products),
                                                "features": len(FEATURES), "mismatches": len(mismatches)}
            examples.extend({"check": "parity_mismatch", **m} for m in mismatches[:50])

            # 7. Missing, stale, latency, outage.
            fresh = client.fetch([missing_product, parity_products[0]], now=end.to_pydatetime())
            stale = client.fetch([parity_products[0]], now=(end + pd.Timedelta(hours=7)).to_pydatetime())
            metrics["missing_and_stale"] = {
                "missing_product_status": fresh["products"][missing_product]["status"],
                "fresh_product_status": fresh["products"][parity_products[0]]["status"],
                "after_7h_status": stale["products"][parity_products[0]]["status"],
                "after_7h_snapshot_age_s": stale["snapshot_age_s"]}
            latencies = []
            for _ in range(500):
                batch = rng.sample(products, 50)
                latencies.append(client.fetch(batch, now=end.to_pydatetime())["fetch_ms"])
            metrics["online_fetch_latency_ms_batch50"] = {
                "n": len(latencies), **{f"p{q}": float(np.percentile(latencies, q)) for q in (50, 95, 99)}}
        finally:
            redis_proc.terminate()
            redis_proc.wait(timeout=10)
        try:
            client.fetch([parity_products[0]], now=end.to_pydatetime())
            metrics["redis_outage"] = {"behavior": "returned values (unexpected)"}
        except FeaturesUnavailable as error:
            metrics["redis_outage"] = {"behavior": "FeaturesUnavailable raised; no values served",
                                       "error": str(error)[:200]}

        (args.evidence / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
        with open(args.evidence / "per_example.jsonl", "w") as handle:
            for example in examples:
                handle.write(json.dumps(example, default=str) + "\n")
        run.manifest["extra"]["exports"] = {tag: {k: v["sha256"] for k, v in e["tables"].items()}
                                            for tag, e in exports.items()}
        print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
