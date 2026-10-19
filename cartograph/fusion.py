"""Hybrid retrieval: BM25 + a dense encoder, fusion chosen on dev, reported once on test.

Candidates: top-100 from each system. Fusions: reciprocal-rank fusion with k in {10, 30, 60, 100}, and weighted
min-max-normalized score fusion with alpha (dense weight) in {0.1, ..., 0.9}. Selection: best dev NDCG@10.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .eval_retrieval import bootstrap, query_metrics, rrf
from .provenance import Run
from .retrieval import SparseBM25
from .vectors import exact_topk, load_slice

DEPTH = 100


def weighted(bm25: list[tuple[str, float]], dense: list[tuple[str, float]], alpha: float) -> list[str]:
    def norm(pairs):
        if not pairs:
            return {}
        values = np.array([s for _, s in pairs])
        low, high = values.min(), values.max()
        return {k: (s - low) / (high - low) if high > low else 1.0 for k, s in pairs}
    b, d = norm(bm25), norm(dense)
    scores = {k: (1 - alpha) * b.get(k, 0.0) + alpha * d.get(k, 0.0) for k in set(b) | set(d)}
    return sorted(scores, key=lambda k: (-scores[k], k))[:DEPTH]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True, help="dir with products.npy, queries_dev/test.npy")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    grid = [("rrf", k) for k in (10, 30, 60, 100)] + [("weighted", a / 10) for a in range(1, 10)]
    config = {"vectors": str(args.vectors), "grid": grid, "depth": DEPTH, "selection": "best dev NDCG@10"}
    with Run(args.evidence, "gate_b_hybrid_fusion", config, inputs={"products.npy": args.vectors / "products.npy"},
             lock=Path("envs/retrieval.lock.txt")) as run:
        products, _, _, _, manifest = load_slice(args.slice, "test")
        ids = products.product_id.to_numpy().astype(str)
        start = time.perf_counter()
        bm25 = SparseBM25(dict(zip(products.product_id, products.text)))
        run.manifest["extra"]["bm25_build_s"] = time.perf_counter() - start
        p = np.load(args.vectors / "products.npy").astype(np.float32)
        results, per_split = {}, {}
        for split in ("dev", "test"):
            _, queries, labels, _, _ = load_slice(args.slice, split)
            q = np.load(args.vectors / f"queries_{split}.npy").astype(np.float32)
            top = exact_topk(p, q, DEPTH)
            scores = (q[:, None, :] * p[top]).sum(-1)
            rows = {name: [] for name in ["bm25", "dense"] + [f"{m}_{v}" for m, v in grid]}
            for i, query_id in enumerate(queries.query_id):
                bm = bm25.search(queries["query"].iloc[i], DEPTH)
                de = list(zip(ids[top[i]].tolist(), scores[i].tolist()))
                judged = labels[query_id]
                rows["bm25"].append(query_metrics([k for k, _ in bm], judged))
                rows["dense"].append(query_metrics([k for k, _ in de], judged))
                for method, value in grid:
                    fused = rrf([[k for k, _ in bm], [k for k, _ in de]], DEPTH, k=int(value)) if method == "rrf" else weighted(bm, de, value)
                    rows[f"{method}_{value}"].append(query_metrics(fused, judged))
            per_split[split] = rows
            results[split] = {name: float(np.mean([r["ndcg@10"] for r in v if r["ndcg@10"] is not None]))
                              for name, v in rows.items()}
        best = max((k for k in results["dev"] if k not in ("bm25", "dense")), key=results["dev"].get)
        test_rows = per_split["test"]
        comparison = {m: bootstrap([{s: test_rows[s][i] for s in ("dense", "bm25", best)} for i in range(len(test_rows["dense"]))],
                                   m, ["dense", "bm25", best], "dense", 1000, 13) for m in ("ndcg@10", "recall@50", "judged@10")}
        metrics = {"dev_ndcg@10": results["dev"], "test_ndcg@10": results["test"], "selected": best,
                   "test_selected_vs_dense_and_bm25": comparison}
        (args.evidence / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        with open(args.evidence / "per_example.jsonl", "w") as handle:
            for split, rows in per_split.items():
                for name in ("bm25", "dense", best):
                    handle.write(json.dumps({"split": split, "system": name, "rows": rows[name]}) + "\n")
        run.manifest["extra"]["slice_id"] = manifest["slice_id"]
        print(json.dumps({"selected": best, "dev": results["dev"][best], "test": results["test"][best],
                          "test_dense": results["test"]["dense"], "test_bm25": results["test"]["bm25"]}))


if __name__ == "__main__":
    main()
