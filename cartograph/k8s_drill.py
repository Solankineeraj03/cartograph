"""Client side of the KubeRay drill (stdlib only, runs on the Mac against a port-forwarded RayService)."""

import argparse
import json
import time
import urllib.request
from pathlib import Path

from .gate_g_drill import REQUESTS


def call(url, body=None, timeout=10):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except Exception as error:
        return -1, {"error": f"{type(error).__name__}: {error}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=["replay", "watch"])
    parser.add_argument("--url", default="http://127.0.0.1:18000")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="host drill traces.jsonl (original_v1)")
    parser.add_argument("--seconds", type=int, default=300)
    args = parser.parse_args()
    if args.cmd == "replay":
        reference = {}
        if args.reference and args.reference.exists():
            for line in open(args.reference):
                t = json.loads(line)
                if t["stage"] == "original_v1":
                    from .release import trace_digest
                    reference[t["request"]] = trace_digest(t) if t.get("decision") != "invalid_request" else None
        rows = []
        for name, text in REQUESTS:
            status, trace = call(f"{args.url}/v1/bundle", {"request": text, "as_of": "2026-09-20T12:00:00+00:00"})
            rows.append({"request": name, "status": status, "release": trace.get("release_id"), "decision": trace.get("decision"),
                         "digest": trace.get("digest"), "host_digest": reference.get(name),
                         "matches_host": reference.get(name) == trace.get("digest") if reference else None,
                         "pod": trace.get("served_by_pod"), "total_ms": trace.get("timings_ms", {}).get("total")})
        args.out.write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(rows, indent=1))
    else:  # one request per second; records failures and the recovery point
        rows, start = [], time.time()
        while time.time() - start < args.seconds:
            t0 = time.time()
            status, trace = call(f"{args.url}/v1/bundle", {"request": REQUESTS[0][1]}, timeout=5)
            rows.append({"t": round(t0 - start, 2), "status": status, "pod": trace.get("served_by_pod"),
                         "release": trace.get("release_id"), "error": trace.get("error")})
            if args.out.with_suffix(".stop").exists():
                break
            time.sleep(max(0, 1 - (time.time() - t0)))
        args.out.write_text(json.dumps(rows, indent=1) + "\n")
        print(json.dumps({"requests": len(rows), "failures": sum(r["status"] != 200 for r in rows)}))


if __name__ == "__main__":
    main()
