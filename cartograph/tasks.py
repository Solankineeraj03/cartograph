"""Bundle tasks over the synthetic operational catalog, with solver ground truth.

The catalog is split into two disjoint product halves by hash: ``train`` tasks (SFT, DPO, RL rollouts)
draw candidates only from the train half; ``test`` tasks only from the eval half, with a different seed.
Each task has an as-of time; prices and stock come from the exported feature snapshot at that time
(the same point-in-time rule Feast uses). Budgets are drawn around the solver's optimum so some tasks
are infeasible by construction. Adversarial probes (fabricated items, duplicates, false prices,
stale stock) are built in ``probes`` from the same tasks.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .retrieval import SparseBM25
from .solver import Candidate, solve
from .verifier import Product

BUNDLES = [
    ("keyboard", "mouse"), ("keyboard_mouse_combo", "mouse_pad"), ("keyboard", "mouse_pad"),
    ("headphones", "microphone"), ("webcam", "microphone"), ("monitor", "laptop_stand"),
    ("headphones", "webcam"), ("keyboard", "usb_hub"), ("mouse", "mouse_pad"), ("speaker", "webcam"),
]
ADJECTIVES = ["quiet", "wireless", "ergonomic", "compact", "gaming", "bluetooth", "rgb", "usb", "silent", ""]
EVIDENCE_PER_CATEGORY = 8


def half(product_id: str) -> str:
    return "train" if hashlib.sha256(f"task-half:{product_id}".encode()).digest()[0] < 128 else "test"


class OfflineFeatures:
    """As-of lookup over an exported snapshot: newest row with feature_ts <= as_of (Feast semantics, no TTL)."""

    def __init__(self, export_dir: Path):
        self.tables = {}
        for name in ("price", "stock"):
            frame = pd.read_parquet(export_dir / f"{name}.parquet").sort_values(["product_id", "feature_ts"])
            self.tables[name] = {pid: g for pid, g in frame.groupby("product_id")}
        self.manifest = json.loads((export_dir / "export_manifest.json").read_text())

    def get(self, product_id: str, as_of: pd.Timestamp) -> dict | None:
        out = {}
        for name, column in (("price", "price_cents"), ("stock", "in_stock")):
            rows = self.tables[name].get(product_id)
            if rows is None:
                return None
            rows = rows[rows.feature_ts <= as_of]
            if rows.empty:
                return None
            row = rows.iloc[-1]
            out[column] = int(row[column]) if column == "price_cents" else bool(row[column])
            out[f"{name}_event_epoch"] = int(row[f"{name}_event_epoch"])
        return out


class Catalog:
    def __init__(self, slice_dir: Path, sim_dir: Path, export_dir: Path):
        products = pd.read_parquet(slice_dir / "products.parquet", columns=["product_id", "product_title", "text"])
        categories = pd.read_parquet(sim_dir / "product_categories.parquet")
        products = products.merge(categories, on="product_id")
        products = products[products.category != "other"].copy()
        products["half"] = products.product_id.map(half)
        self.products = products.set_index("product_id")
        self.features = OfflineFeatures(export_dir)
        self.indexes = {h: SparseBM25(dict(zip(g.product_id, g.product_title.fillna("") + " " + g.category)))
                        for h, g in products.groupby("half")}

    def candidates(self, split: str, category: str, adjective: str, as_of: pd.Timestamp) -> list[dict]:
        query = f"{adjective} {category.replace('_', ' ')}".strip()
        hits = self.indexes[split].search(query, k=200)
        rows = []
        for product_id, score in hits:
            if self.products.loc[product_id, "category"] != category:
                continue
            features = self.features.get(product_id, as_of)
            if features is None:
                continue
            rows.append({"product_id": product_id, "title": str(self.products.loc[product_id, "product_title"])[:120],
                         "category": category, "retrieval_score": round(float(score), 4), **features})
            if len(rows) == EVIDENCE_PER_CATEGORY:
                break
        return rows


def make_task(catalog: Catalog, rng: random.Random, split: str, index: int, as_of_range) -> dict | None:
    categories = list(rng.choice(BUNDLES))
    adjective = rng.choice(ADJECTIVES)
    as_of = as_of_range[0] + (as_of_range[1] - as_of_range[0]) * rng.random()
    as_of = pd.Timestamp(as_of).floor("min")
    evidence = []
    for category in categories:
        evidence += catalog.candidates(split, category, adjective, as_of)
    if not evidence:
        return None
    pool = [Candidate(Product(e["product_id"], e["category"], e["price_cents"], e["in_stock"]), e["retrieval_score"])
            for e in evidence]
    unconstrained = solve(pool, categories, 10**9)
    mode = rng.random()
    if unconstrained["feasible"]:
        best_total = unconstrained["bundle"]["total_cents"]
        cheapest = sum(min(p.product.price_cents for p in pool
                           if p.product.category == cat and p.product.in_stock) for cat in categories)
        if mode < 0.2:
            budget = max(100, int(cheapest * rng.uniform(0.5, 0.95)))  # infeasible by budget
        elif mode < 0.6:
            budget = int(rng.uniform(cheapest, best_total))
        else:
            budget = int(best_total * rng.uniform(1.0, 1.5))
    else:
        budget = rng.randint(2000, 30000)
    budget = int(round(budget / 100) * 100)
    solution = solve(pool, categories, budget)
    words = " and ".join(c.replace("_", " ") for c in categories)
    request = (f"Find a {adjective + ' ' if adjective else ''}{words} bundle under ${budget / 100:.2f}. "
               "Only include available products and explain the tradeoffs.")
    return {"task_id": f"{split}-{index:05d}", "split": split, "request": request, "adjective": adjective,
            "required_categories": categories, "budget_cents": budget, "as_of": as_of.isoformat(),
            "evidence": evidence, "solver": solution,
            "feature_export": catalog.features.manifest["tag"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=3000)
    parser.add_argument("--test", type=int, default=500)
    args = parser.parse_args()
    catalog = Catalog(args.slice, args.sim, args.export)
    sim = json.loads((args.sim / "simulation_manifest.json").read_text())
    start = pd.Timestamp(sim["config"]["start"]) + pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=sim["config"]["days"] - 1)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, count, seed in (("train", args.train, 101), ("test", args.test, 202)):
        rng = random.Random(seed)
        tasks, attempts = [], 0
        while len(tasks) < count:
            attempts += 1
            task = make_task(catalog, rng, split, len(tasks), (start, end))
            if task:
                tasks.append(task)
        with open(args.out / f"tasks_{split}.jsonl", "w") as handle:
            for task in tasks:
                handle.write(json.dumps(task) + "\n")
        feasible = np.mean([t["solver"]["feasible"] for t in tasks])
        summary[split] = {"tasks": len(tasks), "attempts": attempts, "seed": seed, "feasible_fraction": float(feasible),
                          "products_in_evidence": len({e["product_id"] for t in tasks for e in t["evidence"]})}
    train_ids = {e["product_id"] for t in map(json.loads, open(args.out / "tasks_train.jsonl")) for e in t["evidence"]}
    test_ids = {e["product_id"] for t in map(json.loads, open(args.out / "tasks_test.jsonl")) for e in t["evidence"]}
    summary["product_overlap_train_test"] = len(train_ids & test_ids)
    summary["export"] = catalog.features.manifest
    (args.out / "tasks_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "export"}, indent=2))


if __name__ == "__main__":
    main()
