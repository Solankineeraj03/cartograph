"""Seeded synthetic operational events: prices, stock, and daily interactions. All rows are synthetic.

ESCI has no prices, stock, or clicks; everything here is generated and tagged ``synthetic=True``.
Events land as Parquet shards partitioned by product bucket and by delivery batch:
batch 1 holds what was available by ``batch_cutoff``; batch 2 holds later deliveries, including
late events whose event_time falls before the cutoff. Planted faults (fractions in SIM_CONFIG):
late deliveries, duplicate deliveries of the same event_id, invalid rows, and a product with no events.
"""

import argparse
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SIMULATOR_VERSION = "ops-sim-v1"
SIM_CONFIG = {
    "start": "2026-09-01T00:00:00+00:00", "days": 30, "batch_cutoff": "2026-09-21T00:00:00+00:00",
    "buckets": 64, "price_changes_per_month": 2.0, "stock_changes_per_month": 1.5,
    "initial_in_stock": 0.85, "delay_median_s": 120, "late_fraction": 0.03,
    "late_delay_hours": [6, 72], "duplicate_fraction": 0.005, "invalid_fraction": 0.001,
    "interaction_active_fraction": 0.2,
}
# Ordered keyword rules on lowercased titles; first match wins. Heuristic, not ground truth.
CATEGORY_RULES = [
    ("keyboard_mouse_combo", r"keyboard (and|&|\+) mouse|mouse (and|&|\+) keyboard|keyboard mouse combo"),
    ("mouse_pad", r"mouse ?pad|mousepad|desk mat"),
    ("keyboard", r"\bkeyboard\b"),
    ("mouse", r"\b(wireless|bluetooth|gaming|optical|ergonomic|computer|usb|silent|vertical) mouse\b|\bmice\b"),
    ("headphones", r"headphone|headset|earbud|earphone"),
    ("monitor", r"\bmonitor\b(?! mount| arm| stand)"),
    ("webcam", r"webcam|web cam"),
    ("usb_hub", r"usb (c )?hub|docking station"),
    ("laptop_stand", r"laptop stand|monitor stand|monitor arm"),
    ("speaker", r"\bspeakers?\b"),
    ("microphone", r"microphone|\bmic\b"),
    ("charger", r"\bcharger\b|charging cable|power bank"),
]
PRICE_MEDIAN_CENTS = {"keyboard_mouse_combo": 4500, "mouse_pad": 1500, "keyboard": 4500, "mouse": 2500,
                      "headphones": 6000, "monitor": 18000, "webcam": 5000, "usb_hub": 3000,
                      "laptop_stand": 3500, "speaker": 5000, "microphone": 6000, "charger": 2000,
                      "other": 2500}


def category(title: str) -> str:
    lowered = (title or "").casefold()
    for name, pattern in CATEGORY_RULES:
        if re.search(pattern, lowered):
            return name
    return "other"


def bucket(product_id: str, buckets: int) -> int:
    return int.from_bytes(hashlib.sha256(product_id.encode()).digest()[:4], "big") % buckets


def simulate_bucket(products: pd.DataFrame, seed: int, b: int, config: dict) -> pd.DataFrame:
    rng = np.random.default_rng([seed, b])
    start = pd.Timestamp(config["start"])
    horizon = config["days"] * 86400
    rows = []
    for product_id, cat in zip(products.product_id, products.category):
        base = PRICE_MEDIAN_CENTS[cat] * float(np.exp(rng.normal(0, 0.5)))
        price = max(99, int(round(base / 100.0)) * 100 - 1)
        rows.append((product_id, 0.0, "price", {"price_cents": price}))
        in_stock = bool(rng.random() < config["initial_in_stock"])
        rows.append((product_id, 0.0, "stock", {"in_stock": in_stock, "quantity": int(rng.integers(1, 200)) if in_stock else 0}))
        for t in np.sort(rng.uniform(1, horizon, rng.poisson(config["price_changes_per_month"]))):
            price = max(99, int(round(price * float(np.exp(rng.normal(0, 0.12))) / 100.0)) * 100 - 1)
            rows.append((product_id, float(t), "price", {"price_cents": price}))
        for t in np.sort(rng.uniform(1, horizon, rng.poisson(config["stock_changes_per_month"]))):
            in_stock = not in_stock
            rows.append((product_id, float(t), "stock", {"in_stock": in_stock, "quantity": int(rng.integers(1, 200)) if in_stock else 0}))
        if rng.random() < config["interaction_active_fraction"]:
            rate = float(np.exp(rng.normal(1.0, 1.2)))
            for day in range(config["days"]):
                views = int(rng.poisson(rate))
                if views:
                    carts = int(rng.binomial(views, 0.08))
                    rows.append((product_id, day * 86400 + 86399.0, "interaction", {"views": views, "add_to_cart": carts}))
    frame = pd.DataFrame(rows, columns=["product_id", "offset_s", "event_type", "payload"])
    n = len(frame)
    delay = rng.lognormal(np.log(config["delay_median_s"]), 0.6, n)
    late = rng.random(n) < config["late_fraction"]
    low, high = config["late_delay_hours"]
    delay[late] = rng.uniform(low * 3600, high * 3600, late.sum())
    # Microsecond precision: Iceberg timestamps are microseconds; never truncate silently downstream.
    frame["event_time"] = (start + pd.to_timedelta(frame.offset_s, unit="s")).dt.floor("us")
    frame["available_at"] = (frame.event_time + pd.to_timedelta(delay, unit="s")).dt.floor("us")
    frame["planted_late"] = late
    frame["event_id"] = [hashlib.sha256(f"{SIMULATOR_VERSION}:{seed}:{p}:{t}:{e}".encode()).hexdigest()[:24]
                         for p, t, e in zip(frame.product_id, frame.offset_s, frame.event_type)]
    frame["payload"] = frame.payload.map(lambda p: json.dumps(p, sort_keys=True))
    frame["planted_fault"] = "none"
    invalid = rng.random(n) < config["invalid_fraction"]
    frame.loc[invalid, "payload"] = json.dumps({"price_cents": -1})
    frame.loc[invalid, "event_type"] = "price"
    frame.loc[invalid, "planted_fault"] = "invalid_negative_price"
    duplicates = frame[rng.random(n) < config["duplicate_fraction"]].copy()
    duplicates["planted_fault"] = "duplicate_delivery"
    duplicates["available_at"] = duplicates.available_at + pd.Timedelta(minutes=5)
    frame = pd.concat([frame, duplicates], ignore_index=True)
    frame["synthetic"] = True
    frame["simulator_version"] = SIMULATOR_VERSION
    frame["seed"] = seed
    frame["bucket"] = b
    return frame.drop(columns=["offset_s"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--max-products", type=int, default=None)
    args = parser.parse_args()
    products = pd.read_parquet(args.slice / "products.parquet", columns=["product_id", "product_title"])
    if args.max_products:
        products = products.sample(args.max_products, random_state=args.seed).sort_values("product_id")
    products["category"] = products.product_title.map(category)
    # One planted product with no events, to test missing features downstream.
    no_events = products.product_id.iloc[0]
    products["bucket"] = products.product_id.map(lambda p: bucket(p, SIM_CONFIG["buckets"]))
    cutoff = pd.Timestamp(SIM_CONFIG["batch_cutoff"])
    counts = {"batch_1": 0, "batch_2": 0}
    for b, group in products[products.product_id != no_events].groupby("bucket"):
        events = simulate_bucket(group, args.seed, int(b), SIM_CONFIG)
        for name, part in [("batch_1", events[events.available_at <= cutoff]),
                           ("batch_2", events[events.available_at > cutoff])]:
            directory = args.out / "landing" / name
            directory.mkdir(parents=True, exist_ok=True)
            part.to_parquet(directory / f"bucket={b:02d}.parquet", index=False)
            counts[name] += len(part)
    categories = products[["product_id", "category"]]
    categories.to_parquet(args.out / "product_categories.parquet", index=False)
    manifest = {"simulator_version": SIMULATOR_VERSION, "seed": args.seed, "config": SIM_CONFIG,
                "synthetic": True, "products": len(products), "product_without_events": no_events,
                "events_by_batch": counts, "category_rules": CATEGORY_RULES,
                "category_counts": {k: int(v) for k, v in products.category.value_counts().items()},
                "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    (args.out / "simulation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
