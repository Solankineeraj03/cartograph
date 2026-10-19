import json, random
def load(d):
    return {r["task_id"]: r for r in map(json.loads, open(f"/home/user/cartograph/runs/{d}/per_example.jsonl"))}
import sys
a, b = load(sys.argv[1]), load(sys.argv[2])
ids = sorted(a)
for mode in ("raw", "wrapped"):
    for metric in ("correct",):
        x = [float(a[i][mode][metric]) for i in ids]; y = [float(b[i][mode][metric]) for i in ids]
        d = [q - p for p, q in zip(x, y)]
        rng = random.Random(13); boots = []
        for _ in range(5000):
            s = [d[rng.randrange(len(d))] for _ in d]; boots.append(sum(s) / len(s))
        boots.sort()
        only_b = sum(1 for p, q in zip(x, y) if q > p); only_a = sum(1 for p, q in zip(x, y) if p > q)
        print(mode, "A %.3f B %.3f delta %.3f ci95 [%.3f, %.3f] discordant B-only %d A-only %d" %
              (sum(x)/len(x), sum(y)/len(y), sum(d)/len(d), boots[125], boots[4875], only_b, only_a))
m = json.load(open("/home/user/cartograph/runs/gate_e/sft_merged_test/metrics.json"))
print("sft_merged raw", {k: m["raw"][k] for k in ("correct", "optimal_rate_on_feasible", "false_abstention_on_feasible")}, m["raw"]["error_counts"])
