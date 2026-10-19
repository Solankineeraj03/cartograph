import itertools
import random
import unittest

from cartograph.solver import Candidate, solve
from cartograph.verifier import Product, verify_bundle


def candidate(pid, category, price, stock=True, score=1.0):
    return Candidate(Product(pid, category, price, stock), score)


class SolverTests(unittest.TestCase):
    def test_picks_best_valid_bundle_and_respects_budget(self):
        pool = [candidate("k1", "keyboard", 6000, score=3), candidate("k2", "keyboard", 4000, score=2),
                candidate("m1", "mouse", 5000, score=3), candidate("m2", "mouse", 2000, score=1)]
        result = solve(pool, ["keyboard", "mouse"], 10000)
        self.assertEqual(result["bundle"], {"product_ids": ["k2", "m1"], "total_cents": 9000})

    def test_infeasible_reasons(self):
        pool = [candidate("k1", "keyboard", 6000), candidate("m1", "mouse", 5000, stock=False)]
        self.assertEqual(solve(pool, ["keyboard", "mouse"], 99999)["reasons"], ["none_in_stock:mouse"])
        self.assertEqual(solve(pool, ["keyboard", "webcam"], 99999)["reasons"], ["no_candidate:webcam"])
        pool[1] = candidate("m1", "mouse", 5000)
        result = solve(pool, ["keyboard", "mouse"], 10999)
        self.assertFalse(result["feasible"])
        self.assertEqual(result["cheapest_total_cents"], 11000)

    def test_matches_brute_force_feasibility(self):
        rng = random.Random(5)
        for trial in range(300):
            pool = [candidate(f"p{i}", rng.choice(["a", "b", "c"]), rng.randint(100, 9000),
                              rng.random() < 0.7, rng.random()) for i in range(rng.randint(1, 9))]
            cats, budget = ["a", "b"], rng.randint(500, 12000)
            catalog = {c.product.product_id: c.product for c in pool}
            feasible = any(verify_bundle({"product_ids": list(ids), "total_cents": sum(catalog[i].price_cents for i in ids)},
                                         catalog, budget, set(cats))["valid"]
                           for r in (2,) for ids in itertools.permutations(catalog, r))
            self.assertEqual(solve(pool, cats, budget)["feasible"], feasible, trial)


if __name__ == "__main__":
    unittest.main()
