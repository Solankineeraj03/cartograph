"""Gate G governance drill: verifier probes, release gate rejection, promotion, rollback, and request replay.

1. Probes: for held-out tasks, build adversarial replies (fabricated ID, duplicate, false total, out-of-stock
   substitution, over-budget combination, format-only, bundle for an infeasible task, false abstention).
   Each must be scored incorrect. The rejection rate becomes the release's ``probes_rejected``.
2. Releases bind real artifacts by checksum, with metrics read from measured evidence:
   v1 = b1+PCA-256 retrieval + SFT adapter; a degraded candidate = b1 truncated to 64 dims (must be rejected);
   v2 = v1 with the GRPO adapter (passes the gate).
3. Rollback v2 -> v1 (Gate E found GRPO lowered bundle optimality 0.83 -> 0.59, which the gate didn't check),
   then replay saved requests under the restored release and require identical trace digests.
"""

import argparse
import json
from pathlib import Path

from .assistant import score
from .provenance import Run, sha256_file
from .release import Registry, make_manifest, trace_digest
from .service import answer
from .tasks import Catalog

REQUESTS = [
    ("useful_bundle", "Find a quiet wireless keyboard and mouse bundle under $100. Only include available products and explain the tradeoffs."),
    ("impossible_budget", "Find a wireless keyboard and mouse bundle under $5."),
    ("invalid_request", "I want something nice for my desk."),
    ("other_bundle", "Find a webcam and microphone bundle under $150."),
]


def probes(task: dict) -> dict:
    evidence = task["evidence"]
    solution = task["solver"]
    out = {}
    if solution["feasible"]:
        ids, total = solution["bundle"]["product_ids"], solution["bundle"]["total_cents"]
        out["fabricated_id"] = {"decision": "bundle", "product_ids": [ids[0], "B0FAKE0000"], "total_cents": total, "explanation": ""}
        out["duplicate"] = {"decision": "bundle", "product_ids": [ids[0], ids[0]], "total_cents": total, "explanation": ""}
        out["false_total"] = {"decision": "bundle", "product_ids": ids, "total_cents": max(0, total - 100), "explanation": ""}
        out["false_abstention"] = {"decision": "infeasible", "product_ids": [], "total_cents": 0, "explanation": ""}
        chosen = {e["product_id"]: e for e in evidence if e["product_id"] in ids}
        for position, kept in enumerate(ids):  # swap any chosen item for an out-of-stock one of its category
            swaps = [e for e in evidence if not e["in_stock"] and e["category"] == chosen[kept]["category"]]
            if swaps:
                swap = list(ids)
                swap[position] = swaps[0]["product_id"]
                out["stale_stock"] = {"decision": "bundle", "product_ids": swap, "explanation": "",
                                      "total_cents": total - chosen[kept]["price_cents"] + swaps[0]["price_cents"]}
                break
        by_cat = {}
        for e in evidence:
            if e["in_stock"]:
                by_cat.setdefault(e["category"], []).append(e)
        priciest = [max(by_cat[c], key=lambda e: e["price_cents"]) for c in task["required_categories"] if c in by_cat]
        if len(priciest) == len(task["required_categories"]):
            t = sum(e["price_cents"] for e in priciest)
            if t > task["budget_cents"]:
                out["over_budget"] = {"decision": "bundle", "product_ids": [e["product_id"] for e in priciest],
                                      "total_cents": t, "explanation": ""}
    else:
        first = evidence[0]["product_id"] if evidence else "B0FAKE0000"
        out["bundle_for_infeasible"] = {"decision": "bundle", "product_ids": [first, "B0FAKE0001"], "total_cents": 1, "explanation": ""}
    out["format_only"] = {"decision": "bundle"}
    out["not_json_object"] = None
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True, help="~/cartograph/runs (measured evidence)")
    parser.add_argument("--artifacts", type=Path, required=True, help="~/cartograph (checkpoints, data)")
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    with Run(args.evidence, "gate_g_governance_drill", {k: str(v) for k, v in vars(args).items()},
             lock=Path("envs/retrieval.lock.txt")) as run:
        tasks = [json.loads(line) for line in open(args.tasks)]
        probe_rows = []
        for task in tasks:
            for name, reply in probes(task).items():
                result = score(task, reply)
                probe_rows.append({"task_id": task["task_id"], "probe": name, "rejected": not result["correct"],
                                   "errors": result["errors"]})
        rejected = sum(r["rejected"] for r in probe_rows) / len(probe_rows)
        by_probe = {}
        for r in probe_rows:
            by_probe.setdefault(r["probe"], [0, 0])[0 if r["rejected"] else 1] += 1

        def read(path):
            return json.loads((args.runs / path).read_text())
        dims_b1 = read("gate_b/dims_b1_test/metrics.json")["quality"]["ndcg@10"]
        parity = read("gate_c/drills_v5/metrics.json")["online_offline_parity"]["mismatches"]
        sft = read("gate_d/sft_vllm_test/metrics.json")["wrapped"]["correct"]
        grpo = read("gate_e/grpo_rlvr_test/metrics.json")["wrapped"]["correct"]
        export_files = {f"features_{n}": args.export / f"{n}.parquet" for n in ("price", "stock", "popularity")}
        base_files = {"slice_products": args.slice / "products.parquet", "slice_judgments": args.slice / "judgments.parquet",
                      **export_files, "pca256_basis": args.artifacts / "data/vectors/b1_pca256/pca.npz",
                      "encoder_b1_weights": args.artifacts / "checkpoints/encoders/b1_contrastive_mean/model.safetensors"}
        export_manifest = json.loads((args.export / "export_manifest.json").read_text())
        common = {"data_slice": "728e301ef1fb", "feature_export_tag": export_manifest["tag"],
                  "iceberg_snapshots": {k: v["snapshot_id"] for k, v in export_manifest["tables"].items()},
                  "generator_base": "Qwen/Qwen2.5-1.5B-Instruct@989aa798", "verifier": "cartograph/verifier.py"}
        releases = {
            "v1": make_manifest("v1-sft-pca256", {**common, "retrieval": "b1+pca256", "adapter": "sft_full_v1",
                                                  "files": {**base_files, "adapter": args.artifacts / "checkpoints/sft/full_v1/adapter/adapter_model.safetensors"}},
                                {"ndcg@10": dims_b1["pca_256"]["mean"], "task_correct": sft, "parity_mismatches": parity,
                                 "probes_rejected": rejected}),
            "degraded": make_manifest("candidate-trunc64", {**common, "retrieval": "b1+truncate64", "adapter": "sft_full_v1",
                                                            "files": {**base_files, "adapter": args.artifacts / "checkpoints/sft/full_v1/adapter/adapter_model.safetensors"}},
                                      {"ndcg@10": dims_b1["truncate_64"]["mean"], "task_correct": sft, "parity_mismatches": parity,
                                       "probes_rejected": rejected}),
            "v2": make_manifest("v2-grpo-pca256", {**common, "retrieval": "b1+pca256", "adapter": "grpo_rlvr",
                                                   "files": {**base_files, "adapter": args.artifacts / "checkpoints/align/grpo_rlvr/adapter/adapter_model.safetensors"}},
                                {"ndcg@10": dims_b1["pca_256"]["mean"], "task_correct": grpo, "parity_mismatches": parity,
                                 "probes_rejected": rejected}),
        }
        for name, manifest in releases.items():
            (args.evidence / f"release_{name}.json").write_text(json.dumps(manifest, indent=2) + "\n")

        registry = Registry(args.evidence / "registry.json")
        if args.evidence.joinpath("registry.json").exists():
            raise FileExistsError("fresh evidence directory required")
        log = [registry.promote(releases["v1"])]
        catalog = Catalog(args.slice, args.sim, args.export)
        as_of = "2026-09-20T12:00:00+00:00"
        original = {name: answer(text, catalog, as_of, registry.state["active"]) for name, text in REQUESTS}
        log.append(registry.promote(releases["degraded"]))
        log.append(registry.promote(releases["v2"]))
        under_v2 = {name: answer(text, catalog, as_of, registry.state["active"]) for name, text in REQUESTS}
        log.append(registry.rollback())
        replayed = {name: answer(text, catalog, as_of, registry.state["active"]) for name, text in REQUESTS}
        replay = {name: {"original_release": original[name]["release_id"], "replay_release": replayed[name]["release_id"],
                         "decision": replayed[name]["decision"], "digest_match": trace_digest(original[name]) == trace_digest(replayed[name]),
                         "differs_under_v2_release_id": original[name]["release_id"] != under_v2[name]["release_id"]}
                  for name, _ in REQUESTS}
        metrics = {"probes": {"n": len(probe_rows), "rejected_fraction": rejected,
                              "by_probe": {k: {"rejected": v[0], "accepted": v[1]} for k, v in by_probe.items()}},
                   "release_log": log, "replay": replay, "active_after_drill": registry.state["active"],
                   "release_ids": {k: v["release_id"] for k, v in releases.items()}}
        (args.evidence / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
        with open(args.evidence / "traces.jsonl", "w") as handle:
            for stage, traces in (("original_v1", original), ("under_v2", under_v2), ("replay_after_rollback", replayed)):
                for name, trace in traces.items():
                    handle.write(json.dumps({"stage": stage, "request": name, **trace}, default=str) + "\n")
        with open(args.evidence / "per_example.jsonl", "w") as handle:
            for r in probe_rows:
                handle.write(json.dumps(r) + "\n")
        run.manifest["extra"]["release_ids"] = metrics["release_ids"]
        print(json.dumps({k: metrics[k] for k in ("probes", "replay", "active_after_drill")}, indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
