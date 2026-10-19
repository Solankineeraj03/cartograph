"""Encode catalog products and split queries with a pinned or trained encoder into aligned .npy files.

Products follow ``products.parquet`` order; queries follow sorted query_id within each split.
Writes ``<out>/products.npy``, ``<out>/queries_<split>.npy``, and ``<out>/encode_manifest.json``.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import hardware, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--model", required=True, help="HF id (with --revision) or a trained checkpoint dir")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--pooling", default=None, help="override for checkpoints; ST models use their own")
    parser.add_argument("--splits", default="dev,test")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--fp32", action="store_true", help="disable bf16 autocast for trained checkpoints")
    parser.add_argument("--reuse-products", type=Path, default=None, help="existing products.npy to link")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)

    trained = (Path(args.model) / "cartograph_encoder.json").exists()
    if trained:
        from .encoders import Encoder
        encoder = Encoder.load(args.model, args.pooling).to(args.device)
        encode = lambda texts, max_len: encoder.encode(texts, max_len, args.batch_size, bf16=not args.fp32)  # noqa: E731
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.model, revision=args.revision, device=args.device)
        encode = lambda texts, max_len: model.encode(  # noqa: E731
            texts, batch_size=args.batch_size, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False).astype(np.float32)

    args.out.mkdir(parents=True, exist_ok=True)
    info = {"model": args.model, "revision": args.revision, "trained_checkpoint": trained,
            "device": args.device, "fp32": args.fp32, "hardware": hardware(), "timings_s": {}}
    if trained:
        info["checkpoint_config"] = json.loads((Path(args.model) / "cartograph_encoder.json").read_text())
    products = pd.read_parquet(args.slice / "products.parquet", columns=["product_id", "text"])
    target = args.out / "products.npy"
    if args.reuse_products:
        vectors = np.load(args.reuse_products)
        if len(vectors) != len(products):
            raise ValueError("reused product vectors do not match the slice")
        np.save(target, vectors)
        info["products_reused_from"] = str(args.reuse_products)
    elif not target.exists():
        start = time.perf_counter()
        np.save(target, encode(products.text.tolist(), 256))
        info["timings_s"]["products"] = time.perf_counter() - start
    queries = pd.read_parquet(args.slice / "queries.parquet")
    for split in args.splits.split(","):
        subset = queries[queries.split == split].sort_values("query_id")
        start = time.perf_counter()
        np.save(args.out / f"queries_{split}.npy", encode(subset["query"].tolist(), 32))
        info["timings_s"][f"queries_{split}"] = time.perf_counter() - start
    info["sha256"] = {p.name: sha256_file(p) for p in sorted(args.out.glob("*.npy"))}
    info["slice_id"] = json.loads((args.slice / "data_manifest.json").read_text())["slice_id"]
    (args.out / "encode_manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info["timings_s"]))


if __name__ == "__main__":
    main()
