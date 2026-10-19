import math
import unittest

from cartograph.features import latest_available
from cartograph.metrics import evaluate
from cartograph.retrieval import BM25
from cartograph.verifier import Product, verify_bundle


class RetrievalTests(unittest.TestCase):
    def test_exact_term_and_empty_query(self):
        index = BM25({"a": "wireless keyboard", "b": "wired mouse", "c": ""})
        self.assertEqual(index.search("KEYBOARD" )[0][0], "a")
        self.assertEqual(index.search(""), [])

    def test_tie_is_stable(self):
        self.assertEqual([key for key, _ in BM25({"b": "mouse", "a": "mouse"}).search("mouse")], ["a", "b"])


class MetricTests(unittest.TestCase):
    def test_perfect_and_reversed_ranking(self):
        judgments = {"a": "E", "b": "S", "c": "I"}
        self.assertAlmostEqual(evaluate(["a", "b"], judgments, 2)["ndcg"], 1.0)
        expected = (0.1 + 1 / math.log2(3)) / (1 + 0.1 / math.log2(3))
        self.assertAlmostEqual(evaluate(["b", "a"], judgments, 2)["ndcg"], expected)

    def test_missing_and_unjudged(self):
        result = evaluate(["unjudged", "a"], {"a": "E", "b": "S"}, 2)
        self.assertEqual(result["judged_positive_recall"], 0.5)
        self.assertEqual(result["judged_fraction_of_returned"], 0.5)

    def test_undefined_and_invalid(self):
        self.assertIsNone(evaluate([], {"a": "I"})["ndcg"])
        self.assertIsNone(evaluate([], {})["judged_positive_recall"])
        for ranked, labels, k in [(["a", "a"], {"a": "E"}, 2), ([], {"a": "bad"}, 1), ([], {}, 0)]:
            with self.assertRaises(ValueError):
                evaluate(ranked, labels, k)


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.catalog = {"a": Product("a", "keyboard", 4500, True),
                        "b": Product("b", "mouse", 2500, True),
                        "c": Product("c", "mouse", 1500, False)}

    def check(self, ids, total=7000, budget=10000):
        return verify_bundle({"product_ids": ids, "total_cents": total}, self.catalog,
                             budget, {"keyboard", "mouse"})

    def test_valid_at_budget_boundary(self):
        self.assertTrue(self.check(["a", "b"], budget=7000)["valid"])

    def test_reward_hacking_attempts(self):
        for ids, total, error in [(["a", "a"], 9000, "duplicate_product"),
                                  (["a", "unknown"], 7000, "unknown_product"),
                                  (["a", "c"], 6000, "out_of_stock"),
                                  (["a"], 4500, "missing_category"),
                                  (["a", "b"], 1, "incorrect_total")]:
            with self.subTest(error=error):
                result = self.check(ids, total)
                self.assertEqual(result["reward"], 0.0)
                self.assertIn(error, result["errors"])
        self.assertIn("over_budget", self.check(["a", "b"], budget=6999)["errors"])

    def test_malformed_output_never_earns_reward(self):
        for output in [None, [], {}, {"product_ids": [], "total_cents": 0},
                       {"product_ids": ["a"], "total_cents": True},
                       {"product_ids": ["a"], "total_cents": 4500, "extra": "x"}]:
            self.assertEqual(verify_bundle(output, self.catalog, 10000, set())["reward"], 0.0)


class FeatureTests(unittest.TestCase):
    def test_late_and_future_events_excluded(self):
        rows = [
            {"product_id": "a", "event_time": "2026-01-01T09:00:00Z", "available_at": "2026-01-01T09:01:00Z", "value": 1},
            {"product_id": "a", "event_time": "2026-01-01T09:30:00Z", "available_at": "2026-01-01T11:00:00Z", "value": 2},
            {"product_id": "a", "event_time": "2026-01-01T12:00:00Z", "available_at": "2026-01-01T09:00:00Z", "value": 3},
        ]
        self.assertEqual(latest_available(rows, "a", "2026-01-01T10:00:00Z")["value"], 1)
        self.assertEqual(latest_available(rows, "a", "2026-01-01T11:00:00Z")["value"], 2)
        self.assertIsNone(latest_available(rows, "missing", "2026-01-01T10:00:00Z"))

    def test_timezone_and_conflicting_rows(self):
        row = {"product_id": "a", "event_time": "2026-01-01T09:00:00Z", "available_at": "2026-01-01T09:00:00Z", "value": 1}
        self.assertIsNotNone(latest_available([row], "a", "2026-01-01T03:00:00-06:00"))
        with self.assertRaises(ValueError):
            latest_available([row, dict(row, value=2)], "a", "2026-01-01T10:00:00Z")
        with self.assertRaises(ValueError):
            latest_available([], "a", "2026-01-01T10:00:00")


if __name__ == "__main__":
    unittest.main()
