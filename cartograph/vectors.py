"""Shared helpers for dense experiments: slice loading, query encoding, exact search, judged metrics."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .eval_retrieval import bootstrap, query_metrics


def load_slice(slice_dir: Path, split: str):
    products = pd.read_parquet(slice_dir / "products.parquet", columns=["product_id", "text"])
    queries = pd.read_parquet(slice_dir / "queries.parquet")
    judgments = pd.read_parquet(slice_dir / "judgments.parquet")
    queries = queries[queries.split == split].sort_values("query_id").reset_index(drop=True)
    labels = {qid: dict(zip(f.product_id, f.esci_label))
              for qid, f in judgments[judgments.query_id.isin(queries.query_id)].groupby("query_id")}
    manifest = json.loads((slice_dir / "data_manifest.json").read_text())
    return products, queries, labels, judgments, manifest


def train_product_ids(judgments: pd.DataFrame) -> set:
    """Products judged for training queries; the only vectors allowed to fit PCA."""
    return set(judgments.loc[judgments.split == "train", "product_id"])


def exact_topk(matrix: np.ndarray, queries: np.ndarray, k: int, chunk: int = 256) -> np.ndarray:
    import torch
    m = torch.from_numpy(matrix)
    out = np.empty((len(queries), k), dtype=np.int64)
    for start in range(0, len(queries), chunk):
        scores = torch.from_numpy(queries[start:start + chunk]) @ m.T
        out[start:start + chunk] = torch.topk(scores, k, dim=1).indices.numpy()
    return out


def judged_report(ids: np.ndarray, top: np.ndarray, queries: pd.DataFrame, labels: dict) -> list[dict]:
    rows = []
    for i, query_id in enumerate(queries.query_id):
        rows.append(query_metrics(ids[top[i]].tolist(), labels[query_id]))
    return rows


def summarize(per_system: dict[str, list[dict]], baseline: str, resamples: int = 1000, seed: int = 13) -> dict:
    systems = list(per_system)
    rows = [{s: per_system[s][i] for s in systems} for i in range(len(per_system[baseline]))]
    return {metric: bootstrap(rows, metric, systems, baseline, resamples, seed)
            for metric in ("ndcg@10", "recall@50", "judged@10")}
