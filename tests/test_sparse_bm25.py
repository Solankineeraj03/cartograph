import json
import random
import unittest
from pathlib import Path

from cartograph.retrieval import BM25

try:
    from cartograph.retrieval import SparseBM25
    import scipy  # noqa: F401
except ImportError:  # pragma: no cover
    SparseBM25 = None


@unittest.skipIf(SparseBM25 is None, "scipy not installed")
class SparseBM25Tests(unittest.TestCase):
    def assert_same(self, documents, queries, k):
        reference, sparse = BM25(documents), SparseBM25(documents)
        for query in queries:
            expected, actual = reference.search(query, k), sparse.search(query, k)
            self.assertEqual([key for key, _ in expected], [key for key, _ in actual], query)
            for (_, a), (_, b) in zip(expected, actual):
                self.assertAlmostEqual(a, b, places=9)

    def test_matches_reference_on_fixture(self):
        fixture = json.loads((Path(__file__).parent.parent / "data/fixture.json").read_text())
        documents = {p["product_id"]: p["title"] + " " + p["text"] for p in fixture["products"]}
        self.assert_same(documents, [q["text"] for q in fixture["queries"]] + ["", "zzz", "case"], 3)

    def test_matches_reference_on_random_corpus_with_ties(self):
        rng = random.Random(0)
        words = [f"w{i}" for i in range(30)]
        documents = {f"d{i:03d}": " ".join(rng.choices(words, k=rng.randint(0, 12))) for i in range(300)}
        documents.update({"dup-b": "w1 w2", "dup-a": "w1 w2"})
        queries = [" ".join(rng.sample(words, rng.randint(1, 4))) for _ in range(50)] + ["w1 w2"]
        for k in (1, 5, 50, 1000):
            self.assert_same(documents, queries, k)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            SparseBM25({})
        with self.assertRaises(ValueError):
            SparseBM25({"a": "x"}).search("x", 0)


if __name__ == "__main__":
    unittest.main()
