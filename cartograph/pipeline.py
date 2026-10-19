"""Ray Data transforms over synthetic events, committed to Iceberg, exported to Feast Parquet.

Ray Data does the distributed work: validation, event_id dedup, and per-product feature rows,
grouped by product bucket so each group is independent and retry-safe. Iceberg commits happen
on the driver (one writer): events are appended with an anti-join on event_id (replaying a batch
adds nothing), and feature tables are recomputed from all committed events and overwritten. Each
ingest is tagged, and ``export`` writes a tagged snapshot to Parquet with a manifest. This is an
explicit adapter boundary, not a native Feast/Iceberg integration.

Feature-row semantics (latest-value features, price and stock): a row is emitted when an event becomes
available (``feature_ts = available_at``) only if its event_time is newer than every event already
available for that product. A point-in-time join on ``feature_ts`` therefore returns the value of the
newest event known at the cutoff, identical to ``cartograph.features.latest_available``. Late events
superseded by a newer, already-known event never become the served value.
Popularity: a daily 7-day rollup per product, computed at day end from interactions available by
day end + 1 h grace; later arrivals count only toward later windows. Every product with any
interaction known by then gets a row (0 when its window is empty), so a quiet week reads as 0 and
not as an older value.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

EVENT_SCHEMA = pa.schema([
    ("event_id", pa.string()), ("product_id", pa.string()), ("event_type", pa.string()),
    ("event_time", pa.timestamp("us", tz="UTC")), ("available_at", pa.timestamp("us", tz="UTC")),
    ("payload", pa.string()), ("synthetic", pa.bool_()), ("simulator_version", pa.string()),
    ("seed", pa.int64()), ("bucket", pa.int64()), ("ingest_batch", pa.string()),
])
QUARANTINE_SCHEMA = pa.schema(list(EVENT_SCHEMA) + [("reason", pa.string())])
PRICE_SCHEMA = pa.schema([("product_id", pa.string()), ("feature_ts", pa.timestamp("us", tz="UTC")),
                          ("created_ts", pa.timestamp("us", tz="UTC")), ("price_cents", pa.int64()),
                          ("price_event_epoch", pa.int64())])
STOCK_SCHEMA = pa.schema([("product_id", pa.string()), ("feature_ts", pa.timestamp("us", tz="UTC")),
                          ("created_ts", pa.timestamp("us", tz="UTC")), ("in_stock", pa.bool_()),
                          ("quantity", pa.int64()), ("stock_event_epoch", pa.int64())])
POPULARITY_SCHEMA = pa.schema([("product_id", pa.string()), ("feature_ts", pa.timestamp("us", tz="UTC")),
                               ("created_ts", pa.timestamp("us", tz="UTC")), ("views_7d", pa.int64()),
                               ("add_to_cart_7d", pa.int64()), ("popularity_7d", pa.float64()),
                               ("popularity_window_end_epoch", pa.int64())])
GRACE = pd.Timedelta(hours=1)
FAIL_ENV = "CARTOGRAPH_FAIL_BUCKET"
FAIL_MARKER_ENV = "CARTOGRAPH_FAIL_MARKER"


def validate_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and dedupe one bucket. Optionally crashes its worker once (failure drill)."""
    fail_bucket, marker = os.environ.get(FAIL_ENV), os.environ.get(FAIL_MARKER_ENV)
    if fail_bucket is not None and marker and len(frame) and int(frame.bucket.iloc[0]) == int(fail_bucket):
        if not Path(marker).exists():
            Path(marker).write_text(json.dumps({"bucket": int(fail_bucket), "pid": os.getpid()}))
            os._exit(1)  # hard worker death, not a Python exception
    frame = frame.copy()
    frame["event_time"] = pd.to_datetime(frame.event_time, utc=True)
    frame["available_at"] = pd.to_datetime(frame.available_at, utc=True)
    reasons = pd.Series("", index=frame.index)
    payload = frame.payload.map(json.loads)
    price = frame.event_type.eq("price")
    bad_price = price & payload.map(lambda p: type(p.get("price_cents")) is not int or p["price_cents"] < 0)
    reasons[bad_price] = "invalid_price"
    stock = frame.event_type.eq("stock")
    bad_stock = stock & payload.map(lambda p: type(p.get("in_stock")) is not bool or p.get("quantity", 0) < 0)
    reasons[bad_stock] = "invalid_stock"
    reasons[~frame.event_type.isin(["price", "stock", "interaction"])] = "unknown_event_type"
    reasons[frame.available_at < frame.event_time] = "available_before_event"
    reasons[~frame.synthetic.astype(bool)] = "not_marked_synthetic"
    frame["reason"] = reasons
    # Dedupe deliveries: keep the earliest available copy of each event_id.
    frame = frame.sort_values(["event_id", "available_at"]).drop_duplicates("event_id", keep="first")
    return frame.drop(columns=[c for c in ("planted_late", "planted_fault") if c in frame])


def latest_value_rows(events: pd.DataFrame, kind: str) -> pd.DataFrame:
    events = events[events.event_type == kind].sort_values(["product_id", "available_at", "event_time"])
    rows = []
    for product_id, group in events.groupby("product_id", sort=False):
        newest = None
        for event_time, available_at, payload in zip(group.event_time, group.available_at, group.payload):
            if newest is not None and event_time <= newest:
                continue  # superseded or same-time duplicate already known
            newest = event_time
            value = json.loads(payload)
            row = {"product_id": product_id, "feature_ts": available_at, "created_ts": available_at}
            if kind == "price":
                row.update(price_cents=value["price_cents"], price_event_epoch=int(event_time.timestamp()))
            else:
                row.update(in_stock=value["in_stock"], quantity=value["quantity"],
                           stock_event_epoch=int(event_time.timestamp()))
            rows.append(row)
    return pd.DataFrame(rows)


def popularity_rows(events: pd.DataFrame, start: pd.Timestamp, days: int) -> pd.DataFrame:
    events = events[events.event_type == "interaction"]
    if events.empty:
        return pd.DataFrame(columns=POPULARITY_SCHEMA.names)
    payload = events.payload.map(json.loads)
    events = events.assign(views=payload.map(lambda p: p["views"]), carts=payload.map(lambda p: p["add_to_cart"]))
    rows = []
    for day in range(days):
        end = start + pd.Timedelta(days=day + 1)
        window = events[(events.event_time > end - pd.Timedelta(days=7)) & (events.event_time <= end)
                        & (events.available_at <= end + GRACE)]
        seen = events.loc[events.available_at <= end + GRACE, "product_id"].unique()
        totals = window.groupby("product_id")[["views", "carts"]].sum().reindex(seen, fill_value=0)
        for product_id, views, carts in zip(totals.index, totals.views, totals.carts):
            rows.append({"product_id": product_id, "feature_ts": end + GRACE, "created_ts": end + GRACE,
                         "views_7d": int(views), "add_to_cart_7d": int(carts),
                         "popularity_7d": float(np.log1p(views + 5 * carts)),
                         "popularity_window_end_epoch": int(end.timestamp())})
    return pd.DataFrame(rows)


def feature_bucket(frame: pd.DataFrame, start: str, days: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["event_time"] = pd.to_datetime(frame.event_time, utc=True)
    frame["available_at"] = pd.to_datetime(frame.available_at, utc=True)
    parts = []
    for name, rows in [("price", latest_value_rows(frame, "price")), ("stock", latest_value_rows(frame, "stock")),
                       ("popularity", popularity_rows(frame, pd.Timestamp(start), days))]:
        if len(rows):
            parts.append(pd.DataFrame({"table": name, "row": rows.to_json(orient="records", lines=True,
                                                                           date_unit="us").splitlines()}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"table": [], "row": []})


def catalog(warehouse: Path):
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse.mkdir(parents=True, exist_ok=True)
    store = SqlCatalog("cartograph", uri=f"sqlite:///{warehouse}/catalog.db", warehouse=f"file://{warehouse}")
    for namespace in ("ops", "features"):
        store.create_namespace_if_not_exists(namespace)
    return store


def to_arrow(frame: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    frame = frame.copy()
    for field in schema:
        if pa.types.is_timestamp(field.type):
            frame[field.name] = pd.to_datetime(frame[field.name], utc=True)
    return pa.Table.from_pandas(frame[schema.names], schema=schema, preserve_index=False)


def _decode(rows: pd.DataFrame, table: str, schema: pa.Schema) -> pd.DataFrame:
    subset = rows[rows.table == table]
    if subset.empty:
        return pd.DataFrame(columns=schema.names)
    frame = pd.read_json("\n".join(subset.row), lines=True, dtype=False)
    for field in schema:
        if pa.types.is_timestamp(field.type):
            frame[field.name] = pd.to_datetime(frame[field.name], unit="us", utc=True) \
                if pd.api.types.is_numeric_dtype(frame[field.name]) else pd.to_datetime(frame[field.name], utc=True)
    return frame


def ingest(landing: Path, batch: str, warehouse: Path, sim_manifest: dict, tag: str | None,
           ray_parallelism: int | None = None) -> dict:
    import ray

    store = catalog(warehouse)
    events_table = store.create_table_if_not_exists("ops.events", schema=EVENT_SCHEMA)
    quarantine_table = store.create_table_if_not_exists("ops.events_quarantine", schema=QUARANTINE_SCHEMA)

    dataset = ray.data.read_parquet(str(landing / batch))
    validated = dataset.groupby("bucket").map_groups(validate_bucket, batch_format="pandas").to_pandas()
    validated["ingest_batch"] = batch
    good, bad = validated[validated.reason == ""], validated[validated.reason != ""]
    existing = set(events_table.scan(selected_fields=("event_id",)).to_arrow().column("event_id").to_pylist()) \
        if events_table.current_snapshot() else set()
    new_good = good[~good.event_id.isin(existing)]
    if len(new_good):
        events_table.append(to_arrow(new_good, EVENT_SCHEMA))
    existing_bad = set(quarantine_table.scan(selected_fields=("event_id",)).to_arrow().column("event_id").to_pylist()) \
        if quarantine_table.current_snapshot() else set()
    new_bad = bad[~bad.event_id.isin(existing_bad)]
    if len(new_bad):
        quarantine_table.append(to_arrow(new_bad, QUARANTINE_SCHEMA))

    all_events = events_table.scan().to_pandas()
    config = sim_manifest["config"]
    # Features are product-local and buckets partition products, so one block per bucket needs no shuffle.
    # (A single-block groupby over all events stalled for 5.7 h in drills_v2; see failed run notes.)
    blocks = [frame for _, frame in all_events.groupby("bucket")]
    features = ray.data.from_pandas(blocks).map_batches(
        lambda f: feature_bucket(f, config["start"], config["days"]), batch_format="pandas",
        batch_size=None).to_pandas()
    report = {"batch": batch, "read_rows": int(len(validated) + 0), "valid_rows": int(len(good)),
              "quarantined_rows": int(len(bad)), "appended_events": int(len(new_good)),
              "skipped_already_committed": int(len(good) - len(new_good)),
              "appended_quarantine": int(len(new_bad)), "feature_rows": {}}
    for name, schema in [("price", PRICE_SCHEMA), ("stock", STOCK_SCHEMA), ("popularity", POPULARITY_SCHEMA)]:
        table = store.create_table_if_not_exists(f"features.{name}_state", schema=schema)
        frame = _decode(features, name, schema).sort_values(["product_id", "feature_ts"])
        table.overwrite(to_arrow(frame, schema))
        report["feature_rows"][name] = int(len(frame))
    report["snapshots"] = {}
    for identifier in ["ops.events", "ops.events_quarantine", "features.price_state",
                       "features.stock_state", "features.popularity_state"]:
        table = store.load_table(identifier)
        snapshot = table.current_snapshot()
        if tag and snapshot is not None:
            table.manage_snapshots().create_tag(snapshot.snapshot_id, tag).commit()
        report["snapshots"][identifier] = snapshot.snapshot_id if snapshot else None
    return report


def export(warehouse: Path, tag: str, out: Path) -> dict:
    """Write the tagged feature snapshots to immutable Parquet files for Feast's file offline store."""
    from .provenance import sha256_file

    store = catalog(warehouse)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"tag": tag, "tables": {}}
    for name in ("price", "stock", "popularity"):
        table = store.load_table(f"features.{name}_state")
        snapshot = table.snapshot_by_name(tag)
        if snapshot is None:
            raise KeyError(f"No snapshot tagged {tag} for features.{name}_state")
        data = table.scan(snapshot_id=snapshot.snapshot_id).to_arrow()
        path = out / f"{name}.parquet"
        import pyarrow.parquet as pq
        pq.write_table(data, path)
        os.chmod(path, 0o444)
        manifest["tables"][name] = {"iceberg_table": f"features.{name}_state", "snapshot_id": snapshot.snapshot_id,
                                    "rows": data.num_rows, "path": str(path), "sha256": sha256_file(path)}
    events = store.load_table("ops.events").snapshot_by_name(tag)
    manifest["events_snapshot_id"] = events.snapshot_id if events else None
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
