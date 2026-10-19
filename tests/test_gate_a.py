import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import pandas as pd
    import scipy  # noqa: F401
except ImportError:  # pragma: no cover
    pd = None


@unittest.skipIf(pd is None, "pandas/scipy not installed")
class EsciRuleTests(unittest.TestCase):
    def test_normalization_and_grouping(self):
        from cartograph.esci import group_key, normalize
        self.assertEqual(normalize("  Wireless-Keyboard,  QUIET "), "wireless keyboard quiet")
        self.assertEqual(group_key("quiet wireless keyboard"), group_key("Keyboard, wireless quiet quiet"))

    def test_dev_assignment_is_deterministic_and_bounded(self):
        from cartograph.esci import in_dev
        keys = [f"q{i}" for i in range(5000)]
        first = [in_dev(k, 0.1, "s") for k in keys]
        self.assertEqual(first, [in_dev(k, 0.1, "s") for k in keys])
        self.assertTrue(0.08 < sum(first) / len(first) < 0.12)

    def test_audit_counts_duplicates_overlap_and_cold_products(self):
        from cartograph.esci import audit
        frame = pd.DataFrame([
            (1, "a b", "a b", "a b", "p1", "E", "train", "train"),
            (1, "a b", "a b", "a b", "p1", "S", "train", "train"),
            (2, "b a", "b a", "a b", "p2", "I", "test", "test"),
            (3, "c", "c", "c", "p3", "S", "test", "test"),
        ], columns=["query_id", "query", "norm_query", "group_key", "product_id", "esci_label",
                    "official_split", "split"])
        report = audit(frame)
        self.assertEqual(report["duplicate_query_product_rows"], 1)
        self.assertEqual(report["conflicting_query_product_labels"], 1)
        self.assertEqual(report["group_key_in_both_official_splits"], 1)
        self.assertEqual(report["test_products_unseen_in_train"], 2)
        self.assertEqual(report["queries_all_irrelevant"], 1)


@unittest.skipIf(pd is None, "pandas/scipy not installed")
class RetrievalHarnessTests(unittest.TestCase):
    def test_rrf_is_stable(self):
        from cartograph.eval_retrieval import rrf
        self.assertEqual(rrf([["a", "b"], ["b", "a"]], 2), ["a", "b"])
        self.assertEqual(rrf([["a", "b"], ["b", "c"]], 3), ["b", "a", "c"])
        # k changes the winner: "b" is 4th in both lists, "a" is 1st in one.
        # k=1: a = 1/2 > b = 2/5.  k=1000: b = 2/1004 > a = 1/1001.
        lists = [["a", "x", "z", "b"], ["c", "y", "w", "b"]]
        self.assertEqual(rrf(lists, 4, k=1)[0], "a")
        self.assertEqual(rrf(lists, 4, k=1000)[0], "b")

    def test_bm25_harness_on_fixture_slice(self):
        from cartograph import eval_retrieval
        fixture = json.loads((Path(__file__).parent.parent / "data/fixture.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame([{"product_id": p["product_id"], "text": p["title"] + " " + p["text"]}
                          for p in fixture["products"]]).to_parquet(root / "products.parquet")
            pd.DataFrame([{"query_id": i, "query": q["text"], "norm_query": q["text"], "split": "test",
                           "source": "fixture"} for i, q in enumerate(fixture["queries"])]
                         ).to_parquet(root / "queries.parquet")
            pd.DataFrame([{"query_id": i, "product_id": p, "esci_label": l, "split": "test"}
                          for i, q in enumerate(fixture["queries"]) for p, l in q["judgments"].items()]
                         ).to_parquet(root / "judgments.parquet")
            (root / "data_manifest.json").write_text(json.dumps({"slice_id": "fixture"}))
            out = root / "evidence"
            argv = ["x", "--slice", str(root), "--evidence", str(out), "--no-dense", "--bootstrap", "20"]
            with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                eval_retrieval.main()
            metrics = json.loads((out / "metrics.json").read_text())
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(metrics["n_queries"], 2)
            judged = metrics["settings"]["judged_list"]["judged@10"]["bm25"]["mean"]
            self.assertEqual(judged, 1.0)
            self.assertNotEqual(manifest["status"], "failed")
            self.assertIn("per_example.jsonl", manifest["outputs_sha256"])


if __name__ == "__main__":
    unittest.main()
