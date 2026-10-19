"""R7: Qdrant HNSW vs exact search on identical vectors.

For each (m, ef_construct) build, sweep search ``hnsw_ef`` and report:
- ANN recall@10 and @100 against exact top-k (agreement with exact search, not relevance);
- judged NDCG@10 and Recall@50 of the ANN results (relevance, unjudged = zero gain);
- per-query latency through the Python client (sequential, one connection), build time,
  on-disk collection size, and Qdrant resident memory.
Tuning uses dev queries; the chosen setting is then reported once on test.
"""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .provenance import Run, sha256_file
from .vectors import exact_topk, judged_report, load_slice, summarize


def rss_mib(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        return None
    return None


def du_mib(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 2**20


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True, help="encode.py output directory")
    parser.add_argument("--qdrant", type=Path, required=True, help="qdrant binary")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--port", type=int, default=6343)
    parser.add_argument("--builds", default="16:100,32:200")
    parser.add_argument("--efs", default="16,32,64,128,256")
    parser.add_argument("--select-recall", type=float, default=0.95, help="min dev recall@10 for selection")
    parser.add_argument("--max-queries", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    from qdrant_client import QdrantClient, models

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with Run(args.evidence, "gate_b_qdrant_ann", config, seed=args.seed,
             inputs={"products.npy": args.vectors / "products.npy"},
             lock=Path("envs/retrieval.lock.txt")) as run:
        vectors = np.load(args.vectors / "products.npy").astype(np.float32)
        results, per_query = {}, {}
        storage = args.work / "qdrant_storage"
        shutil.rmtree(storage, ignore_errors=True)
        storage.mkdir(parents=True)
        (args.work / "config.yaml").write_text(
            f"storage:\n  storage_path: {storage}\nservice:\n  http_port: {args.port}\n"
            f"  grpc_port: {args.port + 1}\n  host: 127.0.0.1\ntelemetry_disabled: true\n")
        server = subprocess.Popen([str(args.qdrant), "--config-path", str(args.work / "config.yaml")],
                                  stdout=open(args.evidence / "qdrant.log", "w"), stderr=subprocess.STDOUT)
        try:
            client = QdrantClient(host="127.0.0.1", port=args.port, timeout=600)
            for _ in range(60):
                try:
                    client.get_collections()
                    break
                except Exception:
                    time.sleep(1)
            splits = {}
            for split in ("dev", "test"):
                products, queries, labels, _, manifest = load_slice(args.slice, split)
                q = np.load(args.vectors / f"queries_{split}.npy").astype(np.float32)
                if args.max_queries and len(queries) > args.max_queries:
                    keep = np.sort(np.random.default_rng(args.seed).choice(len(queries), args.max_queries, replace=False))
                    queries, q = queries.iloc[keep].reset_index(drop=True), q[keep]
                splits[split] = (queries, labels, q, exact_topk(vectors, q, 100))
            ids = products.product_id.to_numpy().astype(str)
            for split, (queries, labels, q, exact) in splits.items():
                per_query[f"{split}/exact"] = judged_report(ids, exact, queries, labels)

            for build in args.builds.split(","):
                m, ef_construct = (int(x) for x in build.split(":"))
                name = f"m{m}_efc{ef_construct}"
                if client.collection_exists(name):
                    client.delete_collection(name)
                client.create_collection(name, vectors_config=models.VectorParams(
                    size=vectors.shape[1], distance=models.Distance.DOT),
                    hnsw_config=models.HnswConfigDiff(m=m, ef_construct=ef_construct),
                    optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0))
                start = time.perf_counter()
                client.upload_collection(name, vectors=vectors, ids=list(range(len(vectors))),
                                         batch_size=2048, parallel=4)
                client.update_collection(name, optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20000))
                while client.get_collection(name).status != models.CollectionStatus.GREEN:
                    time.sleep(2)
                build_s = time.perf_counter() - start
                info = client.get_collection(name)
                results[name] = {"m": m, "ef_construct": ef_construct, "build_seconds": build_s,
                                 "indexed_vectors": info.indexed_vectors_count,
                                 "disk_mib": du_mib(storage / "collections" / name),
                                 "qdrant_rss_mib": rss_mib(server.pid), "search": {}}
                for ef in [int(x) for x in args.efs.split(",")]:
                    for split, (queries, labels, q, exact) in splits.items():
                        latencies, top = [], np.empty((len(q), 100), dtype=np.int64)
                        for i, vector in enumerate(q):
                            start = time.perf_counter()
                            hits = client.query_points(name, query=vector.tolist(), limit=100,
                                                       search_params=models.SearchParams(hnsw_ef=ef)).points
                            latencies.append(time.perf_counter() - start)
                            top[i] = ([h.id for h in hits] + [-1] * 100)[:100]
                        recall10 = np.mean([len(set(top[i, :10]) & set(exact[i, :10])) / 10 for i in range(len(q))])
                        recall100 = np.mean([len(set(top[i]) & set(exact[i])) / 100 for i in range(len(q))])
                        key = f"{split}/{name}/ef{ef}"
                        per_query[key] = judged_report(ids, np.where(top < 0, 0, top), queries, labels)
                        lat = np.asarray(latencies) * 1000
                        results[name]["search"][f"{split}/ef{ef}"] = {
                            "ann_recall@10_vs_exact": float(recall10), "ann_recall@100_vs_exact": float(recall100),
                            "ndcg@10": float(np.mean([r["ndcg@10"] for r in per_query[key]])),
                            "latency_ms": {"n": len(lat), "p50": float(np.percentile(lat, 50)),
                                           "p95": float(np.percentile(lat, 95)), "p99": float(np.percentile(lat, 99))}}
            # Select on dev: lowest p95 latency with dev recall@10 >= threshold.
            candidates = [(v["search"][f"dev/ef{ef}"]["latency_ms"]["p95"], name, ef)
                          for name, v in results.items() for ef in [int(x) for x in args.efs.split(",")]
                          if v["search"][f"dev/ef{ef}"]["ann_recall@10_vs_exact"] >= args.select_recall]
            selection = None
            if candidates:
                _, name, ef = min(candidates)
                selection = {"collection": name, "hnsw_ef": ef, "rule": f"min dev p95 with recall@10 >= {args.select_recall}",
                             "test": results[name]["search"][f"test/ef{ef}"],
                             "test_quality_vs_exact": summarize(
                                 {"exact": per_query["test/exact"], "ann": per_query[f"test/{name}/ef{ef}"]}, "exact",
                                 seed=args.seed)}
        finally:
            server.terminate()
            server.wait(timeout=30)
        metrics = {"builds": results, "selection": selection,
                   "raw_float32_mib": vectors.nbytes / 2**20, "n_vectors": len(vectors),
                   "queries_per_split": {s: len(v[0]) for s, v in splits.items()},
                   "latency_note": "Python client over HTTP on localhost, sequential, includes client overhead."}
        (args.evidence / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        with open(args.evidence / "per_example.jsonl", "w") as handle:
            for key, rows in per_query.items():
                handle.write(json.dumps({"system": key, "rows": rows}) + "\n")
        run.manifest["extra"].update({"slice_id": manifest["slice_id"],
                                      "vectors_sha256": sha256_file(args.vectors / "products.npy")})
        print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
