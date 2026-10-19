"""Evaluate the served Ray Serve path one request at a time, plus freshness, outage, and replica-failure drills.

Subcommands:
  tasks     Send each held-out task's request text to /v1/bundle (sequentially) at a fresh simulated time; save traces.
  probe     One request at a given simulated time (e.g. a stale one) or after Redis is stopped; save the trace.
  failover  Send sequential requests; after N of them, kill the Generator replica actor; record per-request
            status and the time until the first successful response after the kill.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


def post(url: str, body: dict, timeout: float = 300) -> tuple[int, dict | None, float]:
    data = json.dumps(body).encode()
    request = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read()), time.perf_counter() - start
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.loads(error.read()), time.perf_counter() - start
        except Exception:
            return error.code, None, time.perf_counter() - start
    except Exception as error:  # connection reset during failover, timeouts
        return -1, {"error": f"{type(error).__name__}: {error}"}, time.perf_counter() - start


def cmd_tasks(args):
    tasks = [json.loads(line) for line in open(args.tasks)][:args.max_tasks]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(args.out / "traces.jsonl", "w") as handle:
        for task in tasks:
            status, trace, seconds = post(f"{args.url}/v1/bundle", {"request": task["request"], "now": args.now})
            rows.append({"task_id": task["task_id"], "status": status, "client_ms": seconds * 1000,
                         "decision": (trace or {}).get("decision"), "correct_vs_oracle": (trace or {}).get("correct_vs_oracle"),
                         "oracle_feasible": ((trace or {}).get("oracle") or {}).get("feasible"),
                         "timings_ms": (trace or {}).get("timings_ms", {}),
                         "raw_correct": (((trace or {}).get("attempts") or [{}])[0].get("score") or {}).get("correct")})
            handle.write(json.dumps({"task_id": task["task_id"], "status": status, **(trace or {})}) + "\n")
    stages = sorted({k for r in rows for k in r["timings_ms"] if k != "llm_calls"})
    ok = [r for r in rows if r["status"] == 200 and r["correct_vs_oracle"] is not None]
    summary = {"n": len(rows), "http_200": sum(r["status"] == 200 for r in rows),
               "decisions": {d: sum(r["decision"] == d for r in rows) for d in sorted({str(r["decision"]) for r in rows})},
               "correct_vs_oracle": float(np.mean([r["correct_vs_oracle"] for r in ok])) if ok else None,
               "raw_first_attempt_correct": float(np.mean([bool(r["raw_correct"]) for r in ok])) if ok else None,
               "oracle_feasible_fraction": float(np.mean([bool(r["oracle_feasible"]) for r in ok])) if ok else None,
               "latency_ms": {s: {q: float(np.percentile([r["timings_ms"][s] for r in rows if s in r["timings_ms"]], int(q[1:])))
                                  for q in ("p50", "p95", "p99")} for s in stages + ["client"] if s != "client"},
               "client_ms": {q: float(np.percentile([r["client_ms"] for r in rows], int(q[1:]))) for q in ("p50", "p95", "p99")},
               "note": "Sequential client (one request in flight). The oracle is the exact solver on the same live evidence."}
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    with open(args.out / "per_example.jsonl", "w") as handle:
        for r in rows:
            handle.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2))


def cmd_probe(args):
    status, trace, seconds = post(f"{args.url}/v1/bundle", {"request": args.request, "now": args.now})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"status": status, "client_ms": seconds * 1000, "trace": trace}, indent=2) + "\n")
    print(status, (trace or {}).get("decision"), (trace or {}).get("error") or (trace or {}).get("reason"))


def cmd_failover(args):
    import ray
    ray.init(address="auto", namespace="serve", ignore_reinit_error=True)
    tasks = [json.loads(line) for line in open(args.tasks)][:args.requests]
    rows, killed_at, killed = [], None, None
    for i, task in enumerate(tasks):
        if i == args.kill_after:
            # Serve replicas are named actors; list them through the GCS (no dashboard needed).
            named = [a for a in ray.util.list_named_actors(all_namespaces=True)
                     if "Generator" in a["name"] and "REPLICA" in a["name"].upper()]
            killed = [a["name"] for a in named]
            for a in named:
                ray.kill(ray.get_actor(a["name"], namespace=a["namespace"]), no_restart=True)
            killed_at = time.perf_counter()
        start = time.perf_counter()
        status, trace, seconds = post(f"{args.url}/v1/bundle", {"request": task["request"], "now": args.now}, timeout=600)
        rows.append({"i": i, "t_since_kill_s": (start - killed_at) if killed_at else None, "status": status,
                     "seconds": seconds, "decision": (trace or {}).get("decision"), "error": (trace or {}).get("error")})
    after = [r for r in rows if r["t_since_kill_s"] is not None]
    first_ok = next((r for r in after if r["status"] == 200), None)
    summary = {"requests": len(rows), "kill_after": args.kill_after, "killed_actor_ids": killed,
               "errors_before_kill": sum(r["status"] != 200 for r in rows if r["t_since_kill_s"] is None),
               "errors_after_kill": sum(r["status"] != 200 for r in after),
               "first_success_after_kill_s": (first_ok["t_since_kill_s"] + first_ok["seconds"]) if first_ok else None,
               "note": "Generator replica actor killed with ray.kill(no_restart=True); the Serve controller must start a replacement."}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    with open(args.out / "per_example.jsonl", "w") as handle:
        for r in rows:
            handle.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    for name, fn in (("tasks", cmd_tasks), ("probe", cmd_probe), ("failover", cmd_failover)):
        p = sub.add_parser(name)
        p.set_defaults(func=fn)
        p.add_argument("--url", default="http://127.0.0.1:8012")
        p.add_argument("--now", required=True)
        p.add_argument("--out", type=Path, required=True)
        if name in ("tasks", "failover"):
            p.add_argument("--tasks", type=Path, required=True)
        if name == "tasks":
            p.add_argument("--max-tasks", type=int, default=500)
        if name == "probe":
            p.add_argument("--request", required=True)
        if name == "failover":
            p.add_argument("--requests", type=int, default=40)
            p.add_argument("--kill-after", type=int, default=10)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
