"""R5 dimensionality on one vector set: native vs train-fitted PCA vs naive truncation (control).

PCA is fit only on vectors of products judged for training queries (``split == "train"``), never on
test queries or test-only products. Vectors are re-normalized after projection. Reports exact
full-catalog quality, raw float32 coordinate storage, and exact-search latency per dimension.
Matryoshka-trained prefixes are a separate trained model, evaluated with ``--prefix-only``.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .provenance import Run
from .vectors import exact_topk, judged_report, load_slice, summarize, train_product_ids


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--product-vectors", type=Path, required=True)
    parser.add_argument("--query-vectors", type=Path, required=True, help="aligned to sorted split queries")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dims", default="256,128,64")
    parser.add_argument("--prefix-only", action="store_true", help="Matryoshka model: truncation is the method")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--save-pca", type=int, default=None,
                        help="write train-fitted PCA vectors of this size (products + both splits) to --save-dir")
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with Run(args.evidence, "gate_b_dimensionality", config, seed=args.seed,
             inputs={"product_vectors": args.product_vectors, "query_vectors": args.query_vectors},
             lock=Path("envs/retrieval.lock.txt")) as run:
        products, queries, labels, judgments, manifest = load_slice(args.slice, args.split)
        ids = products.product_id.to_numpy().astype(str)
        p = np.load(args.product_vectors).astype(np.float32)
        q = np.load(args.query_vectors).astype(np.float32)
        if len(p) != len(ids) or len(q) != len(queries):
            raise ValueError("vector files are not aligned with the slice")
        fit_mask = np.isin(ids, list(train_product_ids(judgments)))
        native = p.shape[1]
        variants = {f"native_{native}": (p, q)}
        mean = p[fit_mask].mean(0)
        _, _, vt = np.linalg.svd(p[fit_mask] - mean, full_matrices=False)
        for d in [int(x) for x in args.dims.split(",") if x]:
            variants[f"truncate_{d}"] = (normalize(p[:, :d]), normalize(q[:, :d]))
            if not args.prefix_only:
                basis = vt[:d].T
                variants[f"pca_{d}"] = (normalize((p - mean) @ basis), normalize((q - mean) @ basis))
        if args.save_pca:
            # Same fit as the evaluation (train-judged products only); transforms every split with it.
            args.save_dir.mkdir(parents=True, exist_ok=True)
            basis = vt[:args.save_pca].T
            np.save(args.save_dir / "products.npy", normalize((p - mean) @ basis).astype(np.float32))
            for split in ("dev", "test"):
                raw = np.load(args.query_vectors.parent / f"queries_{split}.npy").astype(np.float32)
                np.save(args.save_dir / f"queries_{split}.npy", normalize((raw - mean) @ basis).astype(np.float32))
            np.savez(args.save_dir / "pca.npz", mean=mean, basis=basis)
            run.manifest["extra"]["saved_pca"] = {"dim": args.save_pca, "dir": str(args.save_dir)}
        per_system, info = {}, {}
        for name, (pv, qv) in variants.items():
            pv, qv = np.ascontiguousarray(pv, dtype=np.float32), np.ascontiguousarray(qv, dtype=np.float32)
            start = time.perf_counter()
            top = exact_topk(pv, qv, 100)
            seconds = time.perf_counter() - start
            per_system[name] = judged_report(ids, top, queries, labels)
            info[name] = {"dim": pv.shape[1], "raw_float32_bytes": int(pv.shape[0] * pv.shape[1] * 4),
                          "exact_search_ms_per_query": seconds * 1000 / len(qv)}
        metrics = {"split": args.split, "n_queries": len(queries), "pca_fit_vectors": int(fit_mask.sum()),
                   "variants": info, "quality": summarize(per_system, f"native_{native}", seed=args.seed),
                   "latency_note": f"exact CPU matmul, {args.threads} threads, batches of 256 queries"}
        (args.evidence / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        with open(args.evidence / "per_example.jsonl", "w") as handle:
            for i, query_id in enumerate(queries.query_id):
                handle.write(json.dumps({"query_id": int(query_id),
                                         **{name: rows[i] for name, rows in per_system.items()}}) + "\n")
        run.manifest["extra"]["slice_id"] = manifest["slice_id"]
        print(json.dumps({k: v["mean"] for k, v in metrics["quality"]["ndcg@10"].items() if k != "n_queries"
                          for v in [v]}, indent=2))


if __name__ == "__main__":
    main()
