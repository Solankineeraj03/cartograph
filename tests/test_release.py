import tempfile
import unittest
from pathlib import Path

from cartograph.release import Registry, gate
from cartograph.service import InvalidRequest, parse_request


def manifest(rid, **metrics):
    return {"release_id": rid, "release": rid, "metrics": metrics}


class ReleaseTests(unittest.TestCase):
    def test_gate_rejects_degraded_and_accepts_within_margin(self):
        base = manifest("a", **{"ndcg@10": 0.40})
        self.assertEqual(gate(manifest("b", **{"ndcg@10": 0.395}), base)["decision"], "accept")
        self.assertEqual(gate(manifest("c", **{"ndcg@10": 0.38}), base)["decision"], "reject")
        self.assertEqual(gate(manifest("d", **{"ndcg@10": 0.41, "parity_mismatches": 3}), base)["decision"], "reject")

    def test_registry_promote_reject_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Registry(Path(tmp) / "r.json")
            registry.promote(manifest("v1", **{"ndcg@10": 0.40}))
            self.assertEqual(registry.promote(manifest("bad", **{"ndcg@10": 0.30}))["decision"], "reject")
            self.assertEqual(registry.state["active"], "v1")
            registry.promote(manifest("v2", **{"ndcg@10": 0.401}))
            self.assertEqual(registry.state["active"], "v2")
            self.assertEqual(registry.rollback()["to"], "v1")
            self.assertEqual(Registry(Path(tmp) / "r.json").state["active"], "v1")
            with self.assertRaises(RuntimeError):
                registry.rollback()

    def test_parse_request(self):
        parsed = parse_request("Find a quiet wireless keyboard and mouse bundle under $100.")
        self.assertEqual(parsed["required_categories"], ["keyboard", "mouse"])
        self.assertEqual(parsed["budget_cents"], 10000)
        self.assertEqual(parsed["adjective"], "quiet wireless")
        with self.assertRaises(InvalidRequest):
            parse_request("a keyboard please")
        # Order of appearance, and combo phrases stay one category (served-path bug found in Gate F).
        self.assertEqual(parse_request("Find a mouse and mouse pad bundle under $40.")["required_categories"],
                         ["mouse", "mouse_pad"])
        self.assertEqual(parse_request("Find a keyboard mouse combo and mouse pad bundle under $60.")["required_categories"],
                         ["keyboard_mouse_combo", "mouse_pad"])
        self.assertEqual(parse_request("Find a monitor and laptop stand bundle under $300.")["required_categories"],
                         ["monitor", "laptop_stand"])


if __name__ == "__main__":
    unittest.main()


class TaskRequestRoundTrip(unittest.TestCase):
    def test_every_bundle_template_parses_back(self):
        from cartograph.tasks import ADJECTIVES, BUNDLES
        for categories in BUNDLES:
            for adjective in ADJECTIVES:
                words = " and ".join(c.replace("_", " ") for c in categories)
                text = f"Find a {adjective + ' ' if adjective else ''}{words} bundle under $99.00."
                with self.subTest(text):
                    self.assertEqual(parse_request(text)["required_categories"], list(categories))


class ProbeTests(unittest.TestCase):
    def test_every_probe_is_rejected(self):
        from cartograph.assistant import score
        from cartograph.gate_g_drill import probes
        from tests.test_assistant import TASK
        built = probes(TASK)
        self.assertIn("stale_stock", built)
        for name, reply in built.items():
            with self.subTest(name):
                self.assertFalse(score(TASK, reply)["correct"])
        infeasible = dict(TASK, budget_cents=5000, solver={"feasible": False, "reasons": ["over_budget"], "bundle": None})
        for name, reply in probes(infeasible).items():
            with self.subTest(name):
                self.assertFalse(score(infeasible, reply)["correct"])
