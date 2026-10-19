"""GPU energy per request from Gate F evidence (board power sampled every 0.5 s while each server ran).

Each `vllm bench serve` result stamps `date` (host local time, CDT −05:00) when the run finishes, and records its
`duration`. The run's window is therefore [date − duration, date]. Energy = trapezoidal integral of `power.draw` over
the window. This is GPU board power only: CPU, memory, fans, and PSU losses are excluded, and idle power is included.

    python3 -m cartograph.energy --out experiments/gate_f/energy.json
"""

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOST_TZ = timezone(timedelta(hours=-5))


def samples(path: Path) -> list[tuple[float, float]]:
    rows = []
    for row in csv.reader(open(path)):
        try:
            rows.append((float(row[0]), float(row[2])))
        except (ValueError, IndexError):
            continue
    return rows


def energy_wh(points: list[tuple[float, float]], start: float, end: float) -> tuple[float, float, int]:
    window = [(t, p) for t, p in points if start <= t <= end]
    if len(window) < 2:
        return float("nan"), float("nan"), len(window)
    joules = sum((t2 - t1) * (p1 + p2) / 2 for (t1, p1), (t2, p2) in zip(window, window[1:]))
    return joules / 3600, joules / (window[-1][0] - window[0][0]), len(window)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence", type=Path, default=ROOT / "experiments/gate_f")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for server in sorted(p for p in args.evidence.iterdir() if (p / "gpu_samples.csv").exists()):
        points = samples(server / "gpu_samples.csv")
        for result in sorted(server.glob("*.json")):
            r = json.loads(result.read_text())
            if "duration" not in r or "date" not in r:
                continue
            end = datetime.strptime(r["date"], "%Y%m%d-%H%M%S").replace(tzinfo=HOST_TZ).timestamp()
            start = end - r["duration"]
            wh, watts, n = energy_wh(points, start, end)
            rows.append({"server": server.name, "run": result.stem, "requests": r["completed"],
                         "output_tokens": r["total_output_tokens"], "duration_s": r["duration"],
                         "mean_gpu_watts": watts, "gpu_wh": wh, "samples": n,
                         "wh_per_request": wh / r["completed"] if r["completed"] else None,
                         "wh_per_1k_output_tokens": wh / r["total_output_tokens"] * 1000 if r["total_output_tokens"] else None})
    args.out.write_text(json.dumps({"note": __doc__.strip().split("\n\n")[1], "runs": rows}, indent=2) + "\n")
    for row in rows:
        if row["run"].startswith(("c", "rate", "bench_c")):
            print(f"{row['server']:20s} {row['run']:10s} {row['mean_gpu_watts']:6.1f} W  {row['wh_per_request']*1000:7.1f} mWh/req  "
                  f"{row['wh_per_1k_output_tokens']:.3f} Wh/1k tok  n={row['samples']}")


if __name__ == "__main__":
    main()
