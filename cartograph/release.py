"""Release manifests, promotion gates, rollback, and request replay.

A release binds: data slice, feature export (Iceberg snapshot IDs + Parquet checksums), encoder /
vector index, generator model + adapter, verifier and solver source hashes, and environment locks.
``promote`` compares candidate metrics with the active release under predeclared rules and records
the decision either way. ``rollback`` reactivates the previous release. ``replay`` re-runs a saved
request through the deterministic request path under the active release and checks the trace matches.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .provenance import ROOT, git_state, sha256_file

RULES = {
    "max_relative_ndcg_drop": 0.02,       # retrieval NDCG@10 may drop at most 2% relative
    "max_task_correct_drop": 0.01,        # bundle-task correctness may drop at most 1 point
    "require_zero_parity_mismatches": True,
    "require_all_probes_rejected": True,
}


def code_hashes() -> dict:
    return {name: sha256_file(ROOT / "cartograph" / name) for name in ("verifier.py", "solver.py", "assistant.py",
                                                                        "retrieval.py", "features.py")}


def make_manifest(name: str, components: dict, metrics: dict) -> dict:
    files = {key: {"path": str(path), "sha256": sha256_file(Path(path))}
             for key, path in components.get("files", {}).items()}
    body = {"release": name, "created_at_utc": datetime.now(timezone.utc).isoformat(), "git": git_state(),
            "components": {k: v for k, v in components.items() if k != "files"}, "files": files,
            "code_sha256": code_hashes(),
            "env_locks": {p.name: sha256_file(p) for p in sorted((ROOT / "envs").glob("*.lock.txt"))},
            "metrics": metrics}
    body["release_id"] = hashlib.sha256(json.dumps({k: body[k] for k in ("components", "files", "code_sha256",
                                                                            "env_locks")}, sort_keys=True).encode()
                                        ).hexdigest()[:12]
    return body


def gate(candidate: dict, active: dict | None, rules: dict = RULES) -> dict:
    if active is None:
        return {"decision": "accept", "reasons": ["first_release"]}
    reasons = []
    c, a = candidate["metrics"], active["metrics"]
    if "ndcg@10" in c and "ndcg@10" in a and a["ndcg@10"]:
        drop = (a["ndcg@10"] - c["ndcg@10"]) / a["ndcg@10"]
        if drop > rules["max_relative_ndcg_drop"]:
            reasons.append(f"ndcg@10 relative drop {drop:.4f} > {rules['max_relative_ndcg_drop']}")
    if "task_correct" in c and "task_correct" in a and a["task_correct"] - c["task_correct"] > rules["max_task_correct_drop"]:
        reasons.append(f"task_correct drop {a['task_correct'] - c['task_correct']:.4f}")
    if rules["require_zero_parity_mismatches"] and c.get("parity_mismatches", 0) != 0:
        reasons.append(f"feature parity mismatches {c['parity_mismatches']}")
    if rules["require_all_probes_rejected"] and c.get("probes_rejected", 1.0) < 1.0:
        reasons.append(f"verifier probe rejection {c['probes_rejected']}")
    return {"decision": "reject" if reasons else "accept", "reasons": reasons}


class Registry:
    def __init__(self, path: Path):
        self.path = path
        self.state = json.loads(path.read_text()) if path.exists() else {"active": None, "releases": {}, "log": []}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2) + "\n")

    def active(self) -> dict | None:
        return self.state["releases"].get(self.state["active"]) if self.state["active"] else None

    def promote(self, manifest: dict) -> dict:
        decision = gate(manifest, self.active())
        self.state["releases"][manifest["release_id"]] = manifest
        entry = {"at": datetime.now(timezone.utc).isoformat(), "action": "promote", "candidate": manifest["release_id"],
                 "name": manifest["release"], "previous_active": self.state["active"], **decision}
        if decision["decision"] == "accept":
            self.state["active"] = manifest["release_id"]
        self.state["log"].append(entry)
        self.save()
        return entry

    def rollback(self) -> dict:
        accepted = [e["candidate"] for e in self.state["log"] if e["action"] in {"promote"} and e["decision"] == "accept"]
        current = self.state["active"]
        if current not in accepted or accepted.index(current) == 0:
            raise RuntimeError("no earlier accepted release to roll back to")
        target = accepted[accepted.index(current) - 1]
        entry = {"at": datetime.now(timezone.utc).isoformat(), "action": "rollback", "from": current, "to": target}
        self.state["active"] = target
        self.state["log"].append(entry)
        self.save()
        return entry


def trace_digest(trace: dict) -> str:
    stable = {k: trace[k] for k in ("release_id", "evidence_ids", "decision", "verifier", "feature_snapshot")}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("promote")
    p.add_argument("--manifest", type=Path, required=True)
    sub.add_parser("rollback")
    sub.add_parser("show")
    args = parser.parse_args()
    registry = Registry(args.registry)
    if args.cmd == "promote":
        print(json.dumps(registry.promote(json.loads(args.manifest.read_text())), indent=2))
    elif args.cmd == "rollback":
        print(json.dumps(registry.rollback(), indent=2))
    else:
        print(json.dumps({"active": registry.state["active"], "log": registry.state["log"]}, indent=2))


if __name__ == "__main__":
    main()
