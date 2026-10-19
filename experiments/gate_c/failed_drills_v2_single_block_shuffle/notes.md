# Failed run: Gate C drills v2

Killed after about 5.7 h. Batch-1 validation (with the injected worker kill) finished in 122 s. Feature computation
was a groupby over `ray.data.from_pandas(all_events)`: one 5M-row block shuffled into 200 partitions on 3 CPUs. It
stayed at `Shuffle 0/1` for the whole time. Fix: one block per product bucket and `map_batches(batch_size=None)`,
with no shuffle, because features are product-local. The log was 2,048 repeated progress blocks; its head and tail are kept.
