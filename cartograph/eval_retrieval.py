"""Gate A retrieval baselines on the audited ESCI slice: BM25, untouched dense encoder, RRF.

Two settings are reported separately:
- judged_list: rank only each query's judged candidates (coverage 100% by construction).
- full_catalog: retrieve top-``depth`` from the whole catalog; unjudged results get zero gain
  and are counted in judged coverage rather than treated as known irrelevant.
"""

import argparse
import gzip
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import GAINS, evaluate
from .provenance import Run
from .retrieval import SparseBM25

RRF_K = 60


def percentiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    array = np.asarray(values) * 1000
    return {"n": len(values), "mean_ms": float(array.mean()),
            **{f"p{q}_ms": float(np.percentile(array, q)) for q in (50, 95, 99)}}


def rrf(rankings: list[list[str]], depth: int, k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking[:depth]):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda key: (-scores[key], key))[:depth]


def top_from_scores(ids: np.ndarray, scores: np.ndarray, depth: int, positive_only: bool) -> list[str]:
    candidates = np.flatnonzero(scores > 0) if positive_only else np.arange(len(scores))
    if len(candidates) > depth:
        candidates = candidates[np.argpartition(-scores[candidates], depth - 1)[:depth]]
    order = np.lexsort((ids[candidates], -scores[candidates]))
    return ids[candidates[order]].tolist()


def query_metrics(ranking: list[str], judgments: dict[str, str]) -> dict:
    at10, at50 = evaluate(ranking, judgments, 10), evaluate(ranking, judgments, 50)
    return {"ndcg@10": at10["ndcg"], "recall@50": at50["judged_positive_recall"],
            "judged@10": at10["judged_fraction_of_returned"],
            "judged@50": at50["judged_fraction_of_returned"], "returned": len(ranking)}


def bootstrap(per_query: list[dict], metric: str, systems: list[str], baseline: str,
              resamples: int, seed: int) -> dict:
    """Mean and paired 95% bootstrap CI over queries where the metric is defined for all systems."""
    rows = [row for row in per_query if all(row[s][metric] is not None for s in systems)]
    rng = random.Random(seed)
    result = {"n_queries": len(rows)}
    if not rows:
        return result
    for system in systems:
        values = [row[system][metric] for row in rows]
        result[system] = {"mean": statistics.fmean(values)}
        draws = []
        for _ in range(resamples):
            sample = [values[rng.randrange(len(values))] for _ in values]
            draws.append(statistics.fmean(sample))
        result[system]["ci95"] = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
        if system != baseline:
            deltas = [row[system][metric] - row[baseline][metric] for row in rows]
            draws = []
            for _ in range(resamples):
                indices = [rng.randrange(len(deltas)) for _ in deltas]
                draws.append(statistics.fmean(deltas[i] for i in indices))
            result[system][f"delta_vs_{baseline}"] = {
                "mean": statistics.fmean(deltas),
                "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}
    return result


def slice_name(row) -> dict:
    tokens = row.norm_query.split()
    return {"source": row.source,
            "query_tokens": "1-2" if len(tokens) <= 2 else "3-4" if len(tokens) <= 4 else "5+",
            "has_digit": str(any(ch.isdigit() for ch in row.norm_query))}


class DenseEncoder:
    def __init__(self, name: str, revision: str, device: str, batch_size: int):
        import torch
        from sentence_transformers import SentenceTransformer

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(name, revision=revision, device=device)

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size or self.batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    def sync(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["test", "dev"])
    parser.add_argument("--max-queries", type=int, default=None, help="Seeded subsample (smoke tests)")
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--encoder-revision", default="1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-cache", type=Path, default=None)
    parser.add_argument("--latency-queries", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    args = parser.parse_args()
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    data_manifest = json.loads((args.slice / "data_manifest.json").read_text())
    config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    config.update({"slice_id": data_manifest["slice_id"], "gain_convention": GAINS,
                   "relevant_for_recall": ["E", "S"], "rrf_k": RRF_K, "bm25": {"k1": 1.5, "b": 0.75}})
    inputs = {name: args.slice / name for name in ("products.parquet", "queries.parquet", "judgments.parquet")}
    with Run(args.evidence, "gate_a_retrieval_baselines", config, inputs=inputs, seed=args.seed,
             lock=Path("envs/retrieval.lock.txt")) as run:
        products = pd.read_parquet(inputs["products.parquet"], columns=["product_id", "text"])
        queries = pd.read_parquet(inputs["queries.parquet"])
        judgments = pd.read_parquet(inputs["judgments.parquet"])
        queries = queries[queries.split == args.split].sort_values("query_id")
        if args.max_queries and len(queries) > args.max_queries:
            queries = queries.sample(args.max_queries, random_state=args.seed).sort_values("query_id")
        labels = {qid: dict(zip(frame.product_id, frame.esci_label))
                  for qid, frame in judgments[judgments.query_id.isin(queries.query_id)].groupby("query_id")}
        ids = products.product_id.to_numpy().astype(str)
        position = {key: i for i, key in enumerate(ids)}
        timings = {}

        start = time.perf_counter()
        bm25 = SparseBM25(dict(zip(products.product_id, products.text)))
        timings["bm25_index_build_s"] = time.perf_counter() - start

        systems = ["bm25"]
        encoder = None
        if not args.no_dense:
            systems += ["dense", "rrf"]
            encoder = DenseEncoder(args.encoder, args.encoder_revision, args.device, args.batch_size)
            cache = args.embedding_cache
            if cache and cache.exists():
                product_vectors = np.load(cache)
                timings["product_embedding_s"] = None
                timings["product_embedding_cache"] = str(cache)
            else:
                encoder.sync()
                start = time.perf_counter()
                product_vectors = encoder.encode(products.text.tolist())
                encoder.sync()
                timings["product_embedding_s"] = time.perf_counter() - start
                timings["product_embedding_per_s"] = len(products) / timings["product_embedding_s"]
                if cache:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    np.save(cache, product_vectors)
            matrix = encoder.torch.from_numpy(product_vectors).to(args.device)
            start = time.perf_counter()
            query_vectors = encoder.encode(queries["query"].tolist())
            timings["query_embedding_batch_s"] = time.perf_counter() - start

        latency = {"bm25_search": [], "dense_query_encode": [], "dense_exact_search": []}
        per_query, outputs = [], []
        for i, row in enumerate(queries.itertuples()):
            judged = labels[row.query_id]
            judged_ids = sorted(judged)
            judged_positions = np.array([position[key] for key in judged_ids])
            record = {"query_id": int(row.query_id), "query": row.query, **slice_name(row),
                      "n_judged": len(judged)}
            output = {"query_id": int(row.query_id), "full_catalog": {}, "judged_list": {}}

            start = time.perf_counter()
            bm25_scores = bm25.scores(row.query)
            bm25_top = top_from_scores(ids, bm25_scores, args.depth, positive_only=True)
            latency["bm25_search"].append(time.perf_counter() - start)
            rankings = {"bm25": (bm25_top, top_from_scores(
                np.array(judged_ids), bm25_scores[judged_positions], len(judged_ids), positive_only=False))}

            if encoder is not None:
                if i < args.latency_queries:
                    encoder.sync()
                    start = time.perf_counter()
                    encoder.encode([row.query], batch_size=1)
                    encoder.sync()
                    latency["dense_query_encode"].append(time.perf_counter() - start)
                vector = encoder.torch.from_numpy(query_vectors[i]).to(args.device)
                encoder.sync()
                start = time.perf_counter()
                scores = matrix @ vector
                top_scores, top_index = encoder.torch.topk(scores, args.depth)
                encoder.sync()
                latency["dense_exact_search"].append(time.perf_counter() - start)
                top_index = top_index.cpu().numpy()
                top_scores = top_scores.cpu().numpy()
                order = np.lexsort((ids[top_index], -top_scores))
                dense_top = ids[top_index[order]].tolist()
                dense_judged = product_vectors[judged_positions] @ query_vectors[i]
                dense_list = top_from_scores(np.array(judged_ids), dense_judged, len(judged_ids), False)
                rankings["dense"] = (dense_top, dense_list)
                rankings["rrf"] = (rrf([bm25_top, dense_top], args.depth),
                                   rrf([rankings["bm25"][1], dense_list], len(judged_ids)))

            for system, (full, judged_rank) in rankings.items():
                record[f"full_catalog/{system}"] = query_metrics(full, judged)
                record[f"judged_list/{system}"] = query_metrics(judged_rank, judged)
                output["full_catalog"][system] = full
                output["judged_list"][system] = judged_rank
            per_query.append(record)
            outputs.append(output)

        metrics = {"split": args.split, "n_queries": len(per_query), "timings": timings,
                   "latency": {key: percentiles(values) for key, values in latency.items()},
                   "latency_note": "BM25 on one CPU thread (scipy); dense on GPU with batch-1 query "
                                   "encoding and exact fp32 matmul search; excludes model load.",
                   "settings": {}, "slices": {}}
        for setting in ("full_catalog", "judged_list"):
            metrics["settings"][setting] = {}
            for metric in ("ndcg@10", "recall@50", "judged@10", "judged@50"):
                rows = [{system: r[f"{setting}/{system}"] for system in systems} for r in per_query]
                metrics["settings"][setting][metric] = bootstrap(
                    rows, metric, systems, "bm25", args.bootstrap, args.seed)
            undefined = sum(r[f"{setting}/bm25"]["ndcg@10"] is None for r in per_query)
            metrics["settings"][setting]["queries_with_undefined_ndcg"] = undefined
        for dimension in ("source", "query_tokens", "has_digit"):
            metrics["slices"][dimension] = {}
            for value in sorted({r[dimension] for r in per_query}):
                subset = [r for r in per_query if r[dimension] == value]
                entry = {"n_queries": len(subset)}
                for system in systems:
                    for setting in ("full_catalog", "judged_list"):
                        values = [r[f"{setting}/{system}"]["ndcg@10"] for r in subset
                                  if r[f"{setting}/{system}"]["ndcg@10"] is not None]
                        entry[f"{setting}/{system}/ndcg@10"] = statistics.fmean(values) if values else None
                metrics["slices"][dimension][value] = entry

        (args.evidence / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        with open(args.evidence / "per_example.jsonl", "w") as handle:
            for record in per_query:
                handle.write(json.dumps(record) + "\n")
        with gzip.open(args.evidence / "rankings.jsonl.gz", "wt") as handle:
            for output in outputs:
                handle.write(json.dumps(output) + "\n")
        run.manifest["extra"]["slice_id"] = data_manifest["slice_id"]
        run.manifest["extra"]["encoder"] = None if args.no_dense else {
            "name": args.encoder, "revision": args.encoder_revision}
        print(json.dumps({s: metrics["settings"][s]["ndcg@10"] for s in metrics["settings"]}, indent=2))


if __name__ == "__main__":
    main()
