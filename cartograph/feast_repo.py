"""Feast definitions over an exported Iceberg snapshot (file offline store) with a Redis online store."""

from datetime import timedelta
from pathlib import Path

FEATURE_VERSION = "features-v1"
# Price and stock rows are emitted only when the value changes, so an unchanged price stays valid.
# A 30-day TTL wrongly expired them (gate_c run 4), and Feast's file offline store then drops those entity rows
# instead of returning nulls. Freshness is enforced by the snapshot watermark in cartograph.online instead.
TTL = {"price": timedelta(days=365), "stock": timedelta(days=365), "popularity": timedelta(days=2)}
FEATURES = [
    "price:price_cents", "price:price_event_epoch",
    "stock:in_stock", "stock:quantity", "stock:stock_event_epoch",
    "popularity:popularity_7d", "popularity:views_7d", "popularity:popularity_window_end_epoch",
]


def write_config(repo: Path, redis_port: int) -> Path:
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "feature_store.yaml").write_text(
        "project: cartograph\n"
        f"registry: {repo / 'registry.db'}\n"
        "provider: local\n"
        "offline_store:\n  type: file\n"
        f"online_store:\n  type: redis\n  connection_string: \"127.0.0.1:{redis_port}\"\n"
        "entity_key_serialization_version: 3\n")
    return repo


def objects(export_dir: Path) -> list:
    from feast import Entity, FeatureView, Field, FileSource
    from feast.types import Bool, Float64, Int64

    product = Entity(name="product", join_keys=["product_id"])

    def source(name):
        return FileSource(name=f"{name}_source", path=str(export_dir / f"{name}.parquet"),
                          timestamp_field="feature_ts", created_timestamp_column="created_ts")

    schemas = {
        "price": [Field(name="price_cents", dtype=Int64), Field(name="price_event_epoch", dtype=Int64)],
        "stock": [Field(name="in_stock", dtype=Bool), Field(name="quantity", dtype=Int64),
                  Field(name="stock_event_epoch", dtype=Int64)],
        "popularity": [Field(name="popularity_7d", dtype=Float64), Field(name="views_7d", dtype=Int64),
                       Field(name="add_to_cart_7d", dtype=Int64),
                       Field(name="popularity_window_end_epoch", dtype=Int64)],
    }
    views = [FeatureView(name=name, entities=[product], ttl=TTL[name], schema=fields, source=source(name),
                         online=True, tags={"feature_version": FEATURE_VERSION, "synthetic": "true"})
             for name, fields in schemas.items()]
    return [product, *views]
