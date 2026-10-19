# Failed run: Gate C drills v3

Failed after 575 s with `KeyError: 'B07PPHGPQN'` in `gate_c.py` line 214, where Feast historical rows were looked up
by (product_id, event_timestamp) in a MultiIndex. Everything before that step completed: three ingests, including
the worker kill and the replay, the exports of ingest_1 and ingest_2, and the schema evolution. Querying Feast
directly for that product afterwards returned correct rows, so the bug is in the harness lookup, not in Feast.
Fix: carry a `row_id` column through `get_historical_features` and match on it; report missing rows as a count.
