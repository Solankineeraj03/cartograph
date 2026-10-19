"""Build a bounded, audited US ESCI slice with leakage-safe splits.

Rules (recorded in the manifest):
- locale == "us" and small_version == 1 (ESCI Task 1 reduced set).
- test = the official test split, untouched.
- Official train queries are grouped by a token-sorted normalized query. Groups that also
  occur in test are excluded from training (``train_excluded_test_overlap``); the rest go to
  ``dev`` (a hashed 10% of groups) or ``train``.
- Catalog = every US product judged for any query in the slice; held-out product text may be
  indexed, held-out judgments are never training signals.
"""

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from .provenance import Run, sha256_file

ESCI_COMMIT = "7916cdf6ab75a462e77f20ab40428a10923998d5"
EXPECTED_SHA256 = {
    "shopping_queries_dataset_examples.parquet": "4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a",
    "shopping_queries_dataset_products.parquet": "25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265",
    "shopping_queries_dataset_sources.csv": "a5fed8ecc016443de40bf3c63098f0e3f23bbe4daa4236f1c38b8c3184778c50",
}
RULES = {
    "locale": "us", "small_version": 1, "dev_fraction": 0.10, "dev_salt": "cartograph-dev-v1",
    "group_key": "nfkc casefold, non-alphanumerics to space, tokens sorted and deduplicated",
    "product_text": "title + ' Brand: ' + brand + ' Color: ' + color + ' ' + bullet points; "
                    "missing fields omitted; truncated to max_text_chars",
    "max_text_chars": 2000,
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    return " ".join(re.sub(r"[^\w]+", " ", text).split())


def group_key(text: str) -> str:
    return " ".join(sorted(set(normalize(text).split())))


def in_dev(key: str, fraction: float, salt: str) -> bool:
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def product_text(row, max_chars: int) -> str:
    parts = [row.product_title or ""]
    if row.product_brand:
        parts.append(f"Brand: {row.product_brand}")
    if row.product_color:
        parts.append(f"Color: {row.product_color}")
    if row.product_bullet_point:
        parts.append(" ".join(row.product_bullet_point.split()))
    return " ".join(p for p in parts if p)[:max_chars]


def audit(examples: pd.DataFrame) -> dict:
    report = {}
    pair = examples.groupby(["query_id", "product_id"]).esci_label
    report["duplicate_query_product_rows"] = int((pair.size() > 1).sum())
    report["conflicting_query_product_labels"] = int((pair.nunique() > 1).sum())
    queries = examples.drop_duplicates("query_id")
    for name, column in [("raw_query", "query"), ("normalized_query", "norm_query"),
                         ("group_key", "group_key")]:
        ids_per = queries.groupby(column).query_id.nunique()
        report[f"{name}_shared_by_multiple_query_ids"] = int((ids_per > 1).sum())
        splits = queries.groupby(column).official_split.nunique()
        report[f"{name}_in_both_official_splits"] = int((splits > 1).sum())
    group_pair = examples.groupby(["group_key", "product_id"]).esci_label.nunique()
    report["group_product_pairs_with_conflicting_labels"] = int((group_pair > 1).sum())
    train_products = set(examples.loc[examples.official_split == "train", "product_id"])
    test = examples[examples.official_split == "test"]
    cold = ~test.product_id.isin(train_products)
    report["test_judgments"] = int(len(test))
    report["test_judgments_on_products_unseen_in_train"] = int(cold.sum())
    report["test_products_unseen_in_train"] = int(test.loc[cold, "product_id"].nunique())
    per_query = examples.groupby("query_id").esci_label
    report["queries_all_irrelevant"] = int(per_query.apply(lambda s: set(s) <= {"I"}).sum())
    report["queries_without_E_or_S"] = int(per_query.apply(lambda s: not set(s) & {"E", "S"}).sum())
    report["label_counts_by_split"] = {
        split: {k: int(v) for k, v in frame.esci_label.value_counts().items()}
        for split, frame in examples.groupby("split")}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    inputs = {name: args.raw / name for name in EXPECTED_SHA256}
    with Run(args.evidence, "gate_a_esci_slice", {"rules": RULES, "esci_commit": ESCI_COMMIT},
             inputs=inputs, lock=Path("envs/retrieval.lock.txt")) as run:
        for name, path in inputs.items():
            if run.manifest["inputs"][name]["sha256"] != EXPECTED_SHA256[name]:
                raise ValueError(f"Checksum mismatch for {name}")
        examples = pd.read_parquet(inputs["shopping_queries_dataset_examples.parquet"], filters=[
            ("product_locale", "==", RULES["locale"]), ("small_version", "==", RULES["small_version"])])
        sources = pd.read_csv(inputs["shopping_queries_dataset_sources.csv"])
        examples = examples.merge(sources, on="query_id", how="left")
        examples["source"] = examples["source"].fillna("unknown")
        examples = examples.rename(columns={"split": "official_split"})
        examples["norm_query"] = examples["query"].map(normalize)
        examples["group_key"] = examples["query"].map(group_key)
        test_groups = set(examples.loc[examples.official_split == "test", "group_key"])

        def assign(row_key, official):
            if official == "test":
                return "test"
            if row_key in test_groups:
                return "train_excluded_test_overlap"
            return "dev" if in_dev(row_key, RULES["dev_fraction"], RULES["dev_salt"]) else "train"

        keys = examples[["group_key", "official_split"]].drop_duplicates()
        keys["split"] = [assign(k, s) for k, s in zip(keys.group_key, keys.official_split)]
        examples = examples.merge(keys, on=["group_key", "official_split"], how="left")
        # Sanity: no group key may land in two training/evaluation splits.
        used = examples[examples.split != "train_excluded_test_overlap"]
        leaking = used.groupby("group_key").split.nunique()
        if (leaking > 1).any():
            raise AssertionError(f"{int((leaking > 1).sum())} group keys span multiple splits")

        catalog_ids = set(examples.product_id)
        products = pd.read_parquet(inputs["shopping_queries_dataset_products.parquet"],
                                   filters=[("product_locale", "==", RULES["locale"])])
        products = products[products.product_id.isin(catalog_ids)].drop_duplicates("product_id")
        missing = catalog_ids - set(products.product_id)
        products["text"] = [product_text(r, RULES["max_text_chars"]) for r in products.itertuples()]
        products = products[["product_id", "product_title", "product_brand", "product_color",
                             "product_bullet_point", "text"]].sort_values("product_id")

        report = audit(examples)
        report["catalog_products"] = int(len(products))
        report["judged_products_missing_from_product_table"] = len(missing)
        report["products_with_empty_title"] = int((products.product_title.fillna("") == "").sum())
        query_table = examples.drop_duplicates("query_id")[
            ["query_id", "query", "norm_query", "group_key", "official_split", "split", "source"]]
        report["queries_by_split"] = {k: int(v) for k, v in query_table.split.value_counts().items()}
        report["judgments_by_split"] = {k: int(v) for k, v in examples.split.value_counts().items()}

        args.out.mkdir(parents=True, exist_ok=True)
        judgments = examples[["example_id", "query_id", "product_id", "esci_label", "split"]]
        judgments.sort_values("example_id").to_parquet(args.out / "judgments.parquet", index=False)
        query_table.sort_values("query_id").to_parquet(args.out / "queries.parquet", index=False)
        products.to_parquet(args.out / "products.parquet", index=False)
        checksums = {p.name: sha256_file(p) for p in sorted(args.out.glob("*.parquet"))}
        slice_id = hashlib.sha256(json.dumps(checksums, sort_keys=True).encode()).hexdigest()[:12]
        data_manifest = {"slice_id": slice_id, "esci_commit": ESCI_COMMIT, "source_license": "Apache-2.0",
                         "source_url": "https://github.com/amazon-science/esci-data",
                         "rules": RULES, "input_sha256": EXPECTED_SHA256, "output_sha256": checksums,
                         "audit": report}
        (args.out / "data_manifest.json").write_text(json.dumps(data_manifest, indent=2) + "\n")
        (args.evidence / "data_manifest.json").write_text(json.dumps(data_manifest, indent=2) + "\n")
        (args.evidence / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        run.manifest["extra"]["slice_id"] = slice_id
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
