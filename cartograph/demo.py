"""Run the synthetic CPU fixture and save its explicitly limited evidence."""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from .features import latest_available
from .metrics import evaluate
from .retrieval import BM25
from .verifier import Product, verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    fixture_path = root / "data/fixture.json"
    raw = fixture_path.read_bytes()
    fixture = json.loads(raw)
    documents = {p["product_id"]: p["title"] + " " + p["text"] for p in fixture["products"]}
    index = BM25(documents)
    queries = []
    for query in fixture["queries"]:
        ranked = index.search(query["text"], k=3)
        queries.append({"query_id": query["query_id"], "results": ranked,
                        "metrics": evaluate([key for key, _ in ranked], query["judgments"], k=3)})
    catalog = {p["product_id"]: Product(**{key: p[key] for key in
               ("product_id", "category", "price_cents", "in_stock")}) for p in fixture["products"]}
    task = fixture["bundle_task"]
    report = {
        "status": "measured_local",
        "scope": "synthetic_cpu_fixture_only",
        "warning": "These fixture scores are not ML quality or GPU benchmark results.",
        "queries": queries,
        "bundle_verification": verify_bundle(task["example_output"], catalog,
                                             task["budget_cents"], set(task["required_categories"])),
        "as_of_feature": latest_available(fixture["feature_events"], "keyboard-a", "2026-01-01T10:00:00Z"),
        "provenance": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.platform(),
            "argv": sys.argv,
            "fixture_sha256": hashlib.sha256(raw).hexdigest(),
            "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted((root / "cartograph").glob("*.py"))},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
