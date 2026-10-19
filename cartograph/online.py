"""Request-time feature client: Feast online reads from Redis plus explicit freshness rules.

It never invents a price or stock value. If Redis is unreachable it raises ``FeaturesUnavailable``.
Products with no materialized row come back ``missing``. When the materialization watermark is
older than ``max_snapshot_age``, every product is ``stale`` and callers must not claim availability.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .feast_repo import FEATURES

WATERMARK_KEY = "cartograph:feature_watermark"


class FeaturesUnavailable(RuntimeError):
    pass


@dataclass
class FeatureClient:
    store: object  # feast.FeatureStore
    redis_port: int
    max_snapshot_age: timedelta = timedelta(hours=6)

    def _redis(self):
        import redis
        return redis.Redis(host="127.0.0.1", port=self.redis_port, socket_timeout=1.0)

    def write_watermark(self, as_of: datetime, export_tag: str, snapshot_ids: dict):
        self._redis().set(WATERMARK_KEY, json.dumps({"as_of": as_of.isoformat(), "export_tag": export_tag,
                                                     "snapshot_ids": snapshot_ids}))

    def fetch(self, product_ids: list[str], now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        start = time.perf_counter()
        try:
            raw = self._redis().get(WATERMARK_KEY)
            values = self.store.get_online_features(
                features=FEATURES, entity_rows=[{"product_id": p} for p in product_ids]).to_dict()
        except Exception as error:  # redis/feast connection errors vary by version
            raise FeaturesUnavailable(f"{type(error).__name__}: {error}") from error
        watermark = json.loads(raw) if raw else None
        as_of = datetime.fromisoformat(watermark["as_of"]) if watermark else None
        snapshot_age = (now - as_of) if as_of else None
        stale = snapshot_age is None or snapshot_age > self.max_snapshot_age
        products = {}
        for i, product_id in enumerate(product_ids):
            price, in_stock = values["price_cents"][i], values["in_stock"][i]
            record = {"price_cents": price, "in_stock": in_stock, "quantity": values["quantity"][i],
                      "popularity_7d": values["popularity_7d"][i]}
            if price is None or in_stock is None:
                record["status"] = "missing"
            elif stale:
                record["status"] = "stale"
            else:
                record["status"] = "ok"
            for name in ("price_event_epoch", "stock_event_epoch"):
                epoch = values[name][i]
                record[name.replace("_epoch", "_age_s")] = (now.timestamp() - epoch) if epoch is not None else None
            products[product_id] = record
        return {"watermark": watermark, "snapshot_age_s": snapshot_age.total_seconds() if snapshot_age else None,
                "stale": stale, "products": products, "fetch_ms": (time.perf_counter() - start) * 1000}
