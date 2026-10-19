import json
import unittest

from cartograph.assistant import parse, run_task, score, solver_reply

TASK = {"task_id": "t", "request": "r", "required_categories": ["keyboard", "mouse"], "budget_cents": 7000,
        "as_of": "2026-09-10T00:00:00+00:00", "feature_export": "ingest_2",
        "evidence": [{"product_id": "k", "category": "keyboard", "price_cents": 4500, "in_stock": True,
                      "title": "K", "retrieval_score": 2.0},
                     {"product_id": "m", "category": "mouse", "price_cents": 2500, "in_stock": True,
                      "title": "M", "retrieval_score": 1.0},
                     {"product_id": "m2", "category": "mouse", "price_cents": 1000, "in_stock": False,
                      "title": "M2", "retrieval_score": 3.0}],
        "solver": {"feasible": True, "bundle": {"product_ids": ["k", "m"], "total_cents": 7000}, "score": 3.0}}


def reply(**kw):
    base = {"decision": "bundle", "product_ids": ["k", "m"], "total_cents": 7000, "explanation": "x"}
    base.update(kw)
    return base


class AssistantTests(unittest.TestCase):
    def test_solver_reply_is_correct(self):
        self.assertTrue(score(TASK, json.loads(solver_reply(TASK)))["correct"])

    def test_probes_fail(self):
        cases = {"fabricated": reply(product_ids=["k", "fake"]), "duplicate": reply(product_ids=["k", "k"]),
                 "false_total": reply(total_cents=100), "stale_stock": reply(product_ids=["k", "m2"], total_cents=5500),
                 "three_items": reply(product_ids=["k", "m", "m2"], total_cents=8000),
                 "false_abstention": {"decision": "infeasible", "product_ids": [], "total_cents": 0, "explanation": ""},
                 "format_only": {"decision": "bundle"}}
        for name, value in cases.items():
            with self.subTest(name):
                self.assertFalse(score(TASK, value)["correct"])

    def test_abstention_only_correct_when_infeasible(self):
        infeasible = dict(TASK, budget_cents=5000,
                          solver={"feasible": False, "reasons": ["over_budget"], "bundle": None})
        abstain = {"decision": "infeasible", "product_ids": [], "total_cents": 0, "explanation": "budget"}
        self.assertTrue(score(infeasible, abstain)["correct"])
        self.assertFalse(score(infeasible, reply())["correct"])

    def test_parse_modes_and_retry(self):
        self.assertEqual(parse('{"a": 1}')[1], "strict")
        self.assertEqual(parse('Sure! {"a": 1} hope it helps')[1], "lenient")
        self.assertEqual(parse("no json")[1], "unparseable")
        replies = iter(["Here: " + json.dumps(reply(total_cents=1)), json.dumps(reply())])
        result = run_task(TASK, lambda history: next(replies))
        self.assertFalse(result["raw"]["correct"])
        self.assertTrue(result["wrapped"]["correct"])
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["shown"], "verified_bundle")


if __name__ == "__main__":
    unittest.main()


class GradedRewardTests(unittest.TestCase):
    def test_graded_reward_orders_answers(self):
        try:
            from cartograph.align import graded_reward
        except ImportError:  # pragma: no cover
            self.skipTest("align deps")
        task = dict(TASK, solver={"feasible": True, "bundle": {"product_ids": ["k", "m"], "total_cents": 7000}, "score": 3.0})
        best = json.dumps(reply())
        self.assertEqual(graded_reward(task, best), 1.0)
        self.assertEqual(graded_reward(task, "```json\n" + best + "\n```"), 0.0)   # not strict
        self.assertEqual(graded_reward(task, json.dumps(reply(total_cents=1))), 0.0)  # invalid
        self.assertEqual(graded_reward(task, json.dumps({"decision": "bundle"})), 0.0)  # format only
        worse = dict(task, evidence=task["evidence"] + [{"product_id": "m3", "category": "mouse", "price_cents": 2000,
                                                          "in_stock": True, "title": "M3", "retrieval_score": 0.5}])
        r = graded_reward(worse, json.dumps(reply(product_ids=["k", "m3"], total_cents=6500)))
        self.assertEqual(r, 0.6)
