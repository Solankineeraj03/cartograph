import json
import random
import unittest

try:
    import pandas as pd
    import pyarrow  # noqa: F401
except ImportError:  # pragma: no cover
    pd = None


@unittest.skipIf(pd is None, "pandas/pyarrow not installed")
class ExportSemanticsTests(unittest.TestCase):
    def test_as_of_lookup_on_export_equals_reference(self):
        from cartograph.features import latest_available
        from cartograph.pipeline import latest_value_rows

        rng = random.Random(3)
        base = pd.Timestamp("2026-09-01T00:00:00Z")
        rows = []
        for i in range(300):
            event = base + pd.Timedelta(minutes=rng.randint(0, 5000))
            delay = pd.Timedelta(minutes=rng.choice([1, 2, 3, 600, 3000]))  # some very late
            rows.append({"product_id": f"p{i % 7}", "event_type": "price", "event_time": event,
                         "available_at": event + delay, "payload": json.dumps({"price_cents": i})})
        events = pd.DataFrame(rows).drop_duplicates(["product_id", "event_time"])
        export = latest_value_rows(events, "price")
        for _ in range(500):
            product = f"p{rng.randrange(7)}"
            cutoff = base + pd.Timedelta(minutes=rng.randint(0, 9000))
            candidates = export[(export.product_id == product) & (export.feature_ts <= cutoff)]
            feast_like = None if candidates.empty else int(candidates.sort_values("feature_ts").iloc[-1].price_cents)
            reference = latest_available(
                [{"product_id": r.product_id, "event_time": r.event_time.isoformat(),
                  "available_at": r.available_at.isoformat(), "payload": r.payload}
                 for r in events[events.product_id == product].itertuples()], product, cutoff.isoformat())
            expected = None if reference is None else json.loads(reference["payload"])["price_cents"]
            self.assertEqual(feast_like, expected, (product, cutoff))

    def test_validation_quarantines_and_dedupes(self):
        from cartograph.pipeline import validate_bucket
        t = pd.Timestamp("2026-09-01T00:00:00Z")
        frame = pd.DataFrame([
            {"event_id": "a", "product_id": "p", "event_type": "price", "event_time": t, "available_at": t,
             "payload": '{"price_cents": 100}', "synthetic": True, "bucket": 0},
            {"event_id": "a", "product_id": "p", "event_type": "price", "event_time": t,
             "available_at": t + pd.Timedelta(minutes=5), "payload": '{"price_cents": 100}', "synthetic": True, "bucket": 0},
            {"event_id": "b", "product_id": "p", "event_type": "price", "event_time": t, "available_at": t,
             "payload": '{"price_cents": -1}', "synthetic": True, "bucket": 0},
            {"event_id": "c", "product_id": "p", "event_type": "stock", "event_time": t,
             "available_at": t - pd.Timedelta(seconds=1), "payload": '{"in_stock": true, "quantity": 1}',
             "synthetic": True, "bucket": 0},
        ])
        out = validate_bucket(frame).set_index("event_id")
        self.assertEqual(len(out), 3)
        self.assertEqual(out.loc["a", "available_at"], t)
        self.assertEqual(out.loc["b", "reason"], "invalid_price")
        self.assertEqual(out.loc["c", "reason"], "available_before_event")

    def test_simulator_is_seeded_and_marked_synthetic(self):
        from cartograph.simulator import SIM_CONFIG, category, simulate_bucket
        products = pd.DataFrame({"product_id": ["x1", "x2", "x3"],
                                 "category": ["keyboard", "mouse", "other"]})
        a, b = simulate_bucket(products, 1, 0, SIM_CONFIG), simulate_bucket(products, 1, 0, SIM_CONFIG)
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue(a.synthetic.all())
        self.assertTrue((a.available_at >= a.event_time).all())
        self.assertEqual(category("Logitech MK270 Wireless Keyboard and Mouse Combo"), "keyboard_mouse_combo")
        self.assertEqual(category("Silent Wireless Mouse, 2.4G"), "mouse")
        self.assertEqual(category("Large Gaming Mouse Pad"), "mouse_pad")


if __name__ == "__main__":
    unittest.main()
