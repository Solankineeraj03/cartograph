"""Materialize an applied Feast repo into Redis up to a cutoff and write the freshness watermark."""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True, help="export dir whose manifest names the snapshots")
    parser.add_argument("--end", required=True, help="ISO cutoff, also the watermark")
    parser.add_argument("--redis-port", type=int, default=6390)
    args = parser.parse_args()
    from feast import FeatureStore

    from .feast_repo import objects
    from .online import FeatureClient
    store = FeatureStore(repo_path=str(args.repo))
    store.apply(objects(args.export))
    end = datetime.fromisoformat(args.end)
    store.materialize(start_date=end - timedelta(days=60), end_date=end)
    manifest = json.loads((args.export / "export_manifest.json").read_text())
    FeatureClient(store, args.redis_port).write_watermark(end, manifest["tag"],
                                                         {k: v["snapshot_id"] for k, v in manifest["tables"].items()})
    print(json.dumps({"materialized_until": args.end, "export_tag": manifest["tag"]}))


if __name__ == "__main__":
    main()
