"""Gate B encoder training: contrastive and graded objectives, pooling, Matryoshka prefixes.

Only ``split == "train"`` judgments are used. Dev queries pick settings; test is evaluated
separately by ``cartograph.eval_dense``.

Objectives:
- contrastive: InfoNCE over in-batch positives plus query-aware hard negatives (the query's own
  judged I/C products). Candidates that are known E/S for the anchor query (other than its target)
  are masked out, so in-batch false negatives are not penalized.
- graded: within-query pairs with label(h) > label(l), loss ``w * softplus(-(s_h - s_l) / tau)``,
  with the predeclared weight ``w = (rank_h - rank_l) / 3`` for ranks E=3, S=2, C=1, I=0.
- contrastive+graded: the sum of both.
Matryoshka: the contrastive term is averaged over normalized prefixes of the given dimensions.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .provenance import Run

RANK = {"E": 3, "S": 2, "C": 1, "I": 0}


class Encoder(torch.nn.Module):
    def __init__(self, path: str, pooling: str, revision: str | None = None):
        super().__init__()
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be mean or cls")
        self.pooling = pooling
        self.tokenizer = AutoTokenizer.from_pretrained(path, revision=revision)
        self.model = AutoModel.from_pretrained(path, revision=revision)

    def forward(self, texts: list[str], max_length: int) -> torch.Tensor:
        device = next(self.model.parameters()).device
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                               return_tensors="pt").to(device)
        hidden = self.model(**batch).last_hidden_state
        if self.pooling == "cls":
            return hidden[:, 0]
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    @torch.no_grad()
    def encode(self, texts: list[str], max_length: int, batch_size: int = 512, bf16: bool = True) -> np.ndarray:
        self.eval()
        order = np.argsort([len(t) for t in texts])  # length bucketing; results restored below
        out = np.empty((len(texts), self.model.config.hidden_size), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            index = order[start:start + batch_size]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and torch.cuda.is_available()):
                vectors = self([texts[i] for i in index], max_length)
            out[index] = F.normalize(vectors.float(), dim=-1).cpu().numpy()
        return out

    def save(self, path: Path, extra: dict):
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "cartograph_encoder.json").write_text(json.dumps({"pooling": self.pooling, **extra}, indent=2))

    @classmethod
    def load(cls, path: str, pooling: str | None = None, revision: str | None = None):
        config_path = Path(path) / "cartograph_encoder.json"
        if pooling is None:
            pooling = json.loads(config_path.read_text())["pooling"] if config_path.exists() else "mean"
        return cls(path, pooling, revision)


def training_queries(slice_dir: Path) -> tuple[list[dict], dict[str, str]]:
    judgments = pd.read_parquet(slice_dir / "judgments.parquet")
    queries = pd.read_parquet(slice_dir / "queries.parquet")
    products = pd.read_parquet(slice_dir / "products.parquet", columns=["product_id", "text"])
    train = judgments[judgments.split == "train"]
    text = dict(zip(products.product_id, products.text))
    query_text = dict(zip(queries.query_id, queries["query"]))
    query_group = dict(zip(queries.query_id, queries.group_key))
    items = []
    for query_id, frame in train.groupby("query_id"):
        labels = dict(zip(frame.product_id, frame.esci_label))
        items.append({"query_id": int(query_id), "query": query_text[query_id],
                      "group": query_group[query_id], "labels": labels,
                      "exact": sorted(p for p, l in labels.items() if l == "E"),
                      "negatives": sorted(p for p, l in labels.items() if l in {"I", "C"})})
    return items, text


def contrastive_loss(q, docs, targets, mask, tau):
    logits = q @ docs.T / tau
    logits = logits.masked_fill(mask, float("-inf"))
    return F.cross_entropy(logits, targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="checkpoint directory")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--base", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--base-revision", default="1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
    parser.add_argument("--pooling", default="mean", choices=["mean", "cls"])
    parser.add_argument("--objective", default="contrastive",
                        choices=["contrastive", "graded", "contrastive+graded"])
    parser.add_argument("--matryoshka", default="", help="comma-separated prefix dims, e.g. 384,256,128,64")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-queries", type=int, default=64)
    parser.add_argument("--graded-products", type=int, default=4, help="judged products per query for pairs")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--query-len", type=int, default=32)
    parser.add_argument("--doc-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda", help="cuda (bf16 autocast) or cpu (fp32)")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cuda = args.device.startswith("cuda")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dims = [int(d) for d in args.matryoshka.split(",") if d]
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config["rank"] = RANK
    data_manifest = json.loads((args.slice / "data_manifest.json").read_text())
    config["slice_id"] = data_manifest["slice_id"]
    with Run(args.evidence, f"gate_b_train_{args.out.name}", config, seed=args.seed,
             inputs={"judgments": args.slice / "judgments.parquet"},
             lock=Path("envs/retrieval.lock.txt")) as run:
        items, text = training_queries(args.slice)
        contrastive_items = [item for item in items if item["exact"]]
        graded_items = [item for item in items if len(set(item["labels"].values())) > 1]
        encoder = Encoder(args.base, args.pooling, args.base_revision).to(args.device)
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=0.01)
        warmup = int(args.steps * args.warmup)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s + 1) / max(1, warmup)
                                                      if s < warmup else (args.steps - s) / max(1, args.steps - warmup))
        rng = random.Random(args.seed)
        log = open(args.evidence / "train_log.jsonl", "w")
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        stats = {"masked_known_positive_entries": 0, "queries_without_hard_negative": 0}
        for step in range(args.steps):
            encoder.train()
            losses = {}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                total = 0.0
                if "contrastive" in args.objective:
                    batch, groups = [], set()
                    while len(batch) < args.batch_queries:
                        item = rng.choice(contrastive_items)
                        if item["group"] not in groups:
                            groups.add(item["group"])
                            batch.append(item)
                    doc_ids = [rng.choice(item["exact"]) for item in batch]
                    for item in batch:
                        if item["negatives"]:
                            doc_ids.append(rng.choice(item["negatives"]))
                        else:
                            stats["queries_without_hard_negative"] += 1
                    q = F.normalize(encoder([i["query"] for i in batch], args.query_len).float(), dim=-1)
                    d = F.normalize(encoder([text[p] for p in doc_ids], args.doc_len).float(), dim=-1)
                    targets = torch.arange(len(batch), device=q.device)
                    mask = torch.zeros(len(batch), len(doc_ids), dtype=torch.bool, device=q.device)
                    for row, item in enumerate(batch):
                        for col, product in enumerate(doc_ids):
                            same_product_elsewhere = product == doc_ids[row] and col != row
                            known_positive = col != row and item["labels"].get(product) in {"E", "S"}
                            if same_product_elsewhere or known_positive:
                                mask[row, col] = True
                    stats["masked_known_positive_entries"] += int(mask.sum())
                    if dims:
                        parts = [contrastive_loss(F.normalize(q[:, :k], dim=-1), F.normalize(d[:, :k], dim=-1),
                                                  targets, mask, args.tau) for k in dims]
                        loss_c = sum(parts) / len(parts)
                        for k, part in zip(dims, parts):
                            losses[f"contrastive@{k}"] = float(part)
                    else:
                        loss_c = contrastive_loss(q, d, targets, mask, args.tau)
                    losses["contrastive"] = float(loss_c)
                    total = total + loss_c
                if "graded" in args.objective:
                    batch = rng.sample(graded_items, min(args.batch_queries // 2, len(graded_items)))
                    queries, docs, pairs = [], [], []
                    for qi, item in enumerate(batch):
                        by_label = {}
                        for product, label in item["labels"].items():
                            by_label.setdefault(label, []).append(product)
                        chosen = [rng.choice(v) for v in by_label.values()]
                        rest = [p for p in item["labels"] if p not in chosen]
                        chosen += rng.sample(rest, max(0, min(len(rest), args.graded_products - len(chosen))))
                        chosen = chosen[:max(args.graded_products, len(by_label))]
                        queries.append(item["query"])
                        base = len(docs)
                        docs.extend(chosen)
                        for a in range(len(chosen)):
                            for b in range(len(chosen)):
                                gap = RANK[item["labels"][chosen[a]]] - RANK[item["labels"][chosen[b]]]
                                if gap > 0:
                                    pairs.append((qi, base + a, base + b, gap / 3))
                    q = F.normalize(encoder(queries, args.query_len).float(), dim=-1)
                    d = F.normalize(encoder([text[p] for p in docs], args.doc_len).float(), dim=-1)
                    qi, hi, lo, w = (torch.tensor(x, device=q.device) for x in zip(*pairs))
                    s_hi = (q[qi] * d[hi]).sum(-1)
                    s_lo = (q[qi] * d[lo]).sum(-1)
                    loss_g = (w.float() * F.softplus(-(s_hi - s_lo) / args.tau)).sum() / w.float().sum()
                    losses["graded"] = float(loss_g)
                    losses["graded_pairs"] = len(pairs)
                    total = total + loss_g
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if not math.isfinite(float(total)):
                raise FloatingPointError(f"non-finite loss at step {step}")
            if step % 25 == 0 or step == args.steps - 1:
                log.write(json.dumps({"step": step, "loss": float(total), **losses,
                                      "lr": scheduler.get_last_lr()[0],
                                      "elapsed_s": time.perf_counter() - start}) + "\n")
                log.flush()
        log.close()
        run.manifest["extra"].update({
            "train_seconds": time.perf_counter() - start,
            "device": args.device,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if cuda else None,
            "contrastive_queries": len(contrastive_items), "graded_queries": len(graded_items),
            **stats, "slice_id": data_manifest["slice_id"]})
        encoder.save(args.out, {"base": args.base, "base_revision": args.base_revision,
                                "objective": args.objective, "matryoshka": dims, "steps": args.steps,
                                "seed": args.seed, "slice_id": data_manifest["slice_id"]})
        print(json.dumps(run.manifest["extra"], indent=2))


if __name__ == "__main__":
    main()
