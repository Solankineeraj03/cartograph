"""Regenerate report figures from committed evidence under experiments/ (no recomputation, no hand-typed numbers).

    python3 -m cartograph.report_figures --out docs/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
E = ROOT / "experiments"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated: categorical slots 1-3, light, all pairs
MARKERS = ["o", "s", "^"]


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def load(path):
    return json.loads((E / path).read_text())


def retrieval_bars(out):
    a = load("gate_a/baselines_test_cpu/metrics.json")["settings"]["full_catalog"]["ndcg@10"]
    b = {n: load(f"gate_b/eval_{n}_test/metrics.json")["quality"]["ndcg@10"]["native_384"]["mean"]
         for n in ("b1_contrastive_mean", "b2_graded_mean", "b3_combined_mean")}
    pca = load("gate_b/dims_b1_test/metrics.json")["quality"]["ndcg@10"]["pca_256"]["mean"]
    rows = [("Frozen MiniLM", a["dense"]["mean"]), ("Graded loss (b2)", b["b2_graded_mean"]),
            ("Contrastive + graded (b3)", b["b3_combined_mean"]), ("BM25", a["bm25"]["mean"]),
            ("b1 + PCA-256", pca), ("Contrastive (b1)", b["b1_contrastive_mean"]),
            ("RRF: BM25 + frozen", a["rrf"]["mean"])]
    fusion = E / "gate_b/fusion_b1_pca256/metrics.json"
    if fusion.exists():
        m = json.loads(fusion.read_text())
        rows.append((f"Hybrid: BM25 + b1/PCA-256 ({m['selected']})", m["test_ndcg@10"][m["selected"]]))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=SERIES[0], height=0.6)
    for i, (_, v) in enumerate(rows):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8, color=INK)
    style(ax, "Full-catalog retrieval, ESCI US test (8,956 queries)", "NDCG@10 (E=1, S=0.1, C=0.01; unjudged = 0)", "")
    ax.set_xlim(0, max(r[1] for r in rows) * 1.15)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "retrieval_ndcg.png", dpi=160)
    plt.close(fig)


def compression(out):
    b1 = load("gate_b/dims_b1_test/metrics.json")
    mrl = load("gate_b/eval_b4_matryoshka_mean_test/metrics.json")
    n = 482105
    mib = lambda d: n * d * 4 / 2**20  # noqa: E731
    series = [("b1 + train-fitted PCA", [(384, b1["quality"]["ndcg@10"]["native_384"]["mean"])] +
               [(d, b1["quality"]["ndcg@10"][f"pca_{d}"]["mean"]) for d in (256, 128, 64)]),
              ("b1 truncated (control)", [(384, b1["quality"]["ndcg@10"]["native_384"]["mean"])] +
               [(d, b1["quality"]["ndcg@10"][f"truncate_{d}"]["mean"]) for d in (256, 128, 64)]),
              ("Matryoshka-trained (b4)", [(384, mrl["quality"]["ndcg@10"]["native_384"]["mean"])] +
               [(d, mrl["quality"]["ndcg@10"][f"truncate_{d}"]["mean"]) for d in (256, 128, 64)])]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, (name, points) in enumerate(series):
        xs, ys = [mib(d) for d, _ in points], [v for _, v in points]
        ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker=MARKERS[i], markersize=6, label=name)
        if i == 0:  # dimension labels once, on the top series
            for (d, _), x, y in zip(points, xs, ys):
                ax.annotate(f"{d} dims", (x, y), xytext=(0, 8), textcoords="offset points", fontsize=7,
                            color=INK2, ha="center")
    style(ax, "Quality vs raw vector storage (test NDCG@10)", "Raw fp32 vector storage, MiB (482,105 products)", "NDCG@10")
    ax.set_xlim(60, 780)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "compression_frontier.png", dpi=160)
    plt.close(fig)


def assistant_bars(out):
    rows = [("Base model", load("gate_d/base_vllm_test/metrics.json")),
            ("SFT (LoRA)", load("gate_d/sft_vllm_test/metrics.json")),
            ("GRPO, binary", load("gate_e/grpo_rlvr_test/metrics.json")),
            ("DPO lr 5e-6", load("gate_e/dpo_verifier_v2_test/metrics.json")),
            ("DPO lr 5e-5 x3", load("gate_e/dpo_verifier_v3_lr5e-5_ep3_test/metrics.json"))]
    graded = E / "gate_e/grpo_graded_test/metrics.json"
    if graded.exists():
        rows.append(("GRPO, graded", json.loads(graded.read_text())))
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    width = 0.36
    for j, (mode, label) in enumerate((("raw", "First reply, strict parse"), ("wrapped", "Verify + one retry"))):
        values = [m[mode]["correct"] for _, m in rows]
        xs = [i + (j - 0.5) * (width + 0.02) for i in range(len(rows))]
        ax.bar(xs, values, width=width, color=SERIES[j], label=label)
        for x, v in zip(xs, values):
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.axhline(1.0, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(len(rows) - 0.55, 1.03, "Exact solver = 1.00", fontsize=7, color=INK2, ha="right")
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    style(ax, "Correct decisions on 500 held-out bundle tasks", "", "Fraction correct")
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(out / "assistant_correctness.png", dpi=160)
    plt.close(fig)


def grpo_curves(out):
    runs = [("Binary reward (round 1)", "gate_e/grpo_rlvr/train_log.jsonl")]
    if (E / "gate_e/grpo_graded/train_log.jsonl").exists():
        runs.append(("Graded reward (round 2)", "gate_e/grpo_graded/train_log.jsonl"))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for i, (name, path) in enumerate(runs):
        rows = [json.loads(line) for line in open(E / path) if '"rewards/' in line]
        key = [k for k in rows[0] if k.startswith("rewards/") and k.endswith("/mean")][0]
        window = 10
        for ax, metric in zip(axes, (key, "frac_reward_zero_std")):
            values = [r[metric] for r in rows]
            smooth = [sum(values[max(0, t - window + 1):t + 1]) / len(values[max(0, t - window + 1):t + 1]) for t in range(len(values))]
            ax.plot(range(len(smooth)), smooth, color=SERIES[i], linewidth=2, label=name)
    style(axes[0], "Training reward (10-step mean)", "Step", "Mean reward")
    style(axes[1], "Groups with identical rewards", "Step", "Fraction of groups")
    axes[1].set_ylim(0, 1.02)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "grpo_training.png", dpi=160)
    plt.close(fig)


def serving(out):
    base = E / "gate_f"
    configs = [("vLLM bf16", "bf16"), ("vLLM FP8 weights", "quant_fp8"), ("vLLM bnb 4-bit", "quant_bitsandbytes")]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    drawn = 0
    for i, (name, d) in enumerate(configs):
        points = []
        for c in (1, 2, 4, 8, 16, 32, 64):
            f = base / d / f"c{c}.json"
            if f.exists():
                r = json.loads(f.read_text())
                points.append((c, r["output_throughput"], r["p95_e2el_ms"] / 1000))
        if not points:
            continue
        ax.plot([p[1] for p in points], [p[2] for p in points], color=SERIES[i], linewidth=2, marker=MARKERS[i],
                markersize=6, label=name)
        c, x, y = points[-1]
        ax.annotate(f"{name}, concurrency {c}", (x, y), xytext=(-8, 6), textcoords="offset points", fontsize=7,
                    color=INK, ha="right")
        drawn += 1
    if not drawn:
        plt.close(fig)
        return
    style(ax, "Throughput vs p95 latency, one RTX 3090 (~1k-token prompts, 128 out)",
          "Output tokens / s", "p95 end-to-end latency, s")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "serving_frontier.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for fn in (retrieval_bars, compression, assistant_bars, grpo_curves, serving):
        fn(args.out)
    print(sorted(p.name for p in args.out.glob("*.png")))


if __name__ == "__main__":
    main()
