# Failed run: Gate C drills v4

Failed after 1,255 s with KeyError in the online/offline parity step (gate_c.py:290). Everything before it completed,
including the historical-vs-reference checks. Cause, verified directly: Feast's file offline store drops entity rows
whose latest feature row is older than the FeatureView TTL, rather than returning nulls. With a 30-day TTL on
change-only price/stock rows, 264 of 2,001 parity products had no price row (262 of them had a latest price older than
30 days; the other 2 had an old row plus one newer than the query time). Redis still served those values online, so this
was a real parity bug, not only a harness bug. Fix: TTL 365 days for change-only features (freshness comes from the
watermark), match rows by row_id, and count missing rows instead of crashing.
