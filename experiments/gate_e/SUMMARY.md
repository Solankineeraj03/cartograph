# Gate E: alignment branches from one SFT checkpoint (RLVR via GRPO, DPO on verifier pairs)

Status: `measured_local` on GPU 1 only. **These are not RLHF.** All preference and reward signals here come from the
deterministic verifier (synthetic). Human-feedback RL still needs owner-labeled pairs (`cartograph/label.py`,
`docs/PREFERENCE_RUBRIC.md`), and the GRPO code refuses a reward model trained on non-human labels.

Branch point: the SFT LoRA (`experiments/gate_d/sft_full`) merged into Qwen2.5-1.5B-Instruct (`sft_merged`).
Every branch adds a new LoRA (r=16) on those merged weights and is scored on the same 500 held-out tasks (disjoint
products) against the merged SFT checkpoint itself (`sft_merged_test`, raw 0.690, matching the unmerged adapter's 0.688).

## Results (500 held-out tasks; paired bootstrap, 5,000 resamples)

| | SFT (merged) | GRPO, verifier reward | DPO, verifier pairs | Exact solver |
|---|---|---|---|---|
| Correct, raw | 0.690 | 0.702 | 0.688 | 1.000 |
| Δ vs SFT, raw (95% CI) | — | +0.012 [−0.022, +0.044] | −0.002 [−0.010, +0.004] | |
| Correct, wrapped (retry) | 0.748 | 0.754 | 0.746 | |
| Over-budget errors, raw | 104 | **83** | 103 | 0 |
| Correct abstention, infeasible, raw | 0.80 | **0.86** | 0.80 | 1.00 |
| False abstention, feasible, raw | 0.027 | 0.060 | 0.032 | 0 |
| **Optimal among valid bundles, raw** | **0.831** | **0.593** | 0.827 | 1.000 |

Neither branch improves held-out correctness significantly.

## RLVR (GRPO, `grpo_rlvr`)
TRL 0.24 GRPO with a new LoRA, 300 steps, 4 samples per prompt, temperature 1.0, β (KL) 0.04, lr 1e-5, prompts up to
1,280 tokens (max seen 1,197; the run refuses truncation), completions up to 384 tokens. Reward = 1 only for a
strictly parsed, verifier-correct decision; format alone earns 0. 4,598 s, 11.4 GiB peak reserved.

- Training reward rose from 0.80 (steps 0–30) to 0.90 (steps 200–300). KL from the reference stayed around 0.003 and
  token entropy around 0.016. **1,061 of 1,200 groups (88%) had identical rewards across all four samples** and gave
  no gradient. The SFT policy is near-deterministic (SFT loss 0.013), so GRPO had little to compare.
- Held out, GRPO moved behavior where the reward pays: 21 fewer over-budget bundles and more correct abstentions.
  Net correctness change isn't significant.
- **Reward hacking toward cheap, valid bundles:** the binary reward doesn't distinguish the best valid bundle from a
  worse one, so optimality among valid bundles dropped from 0.83 to 0.59. A real regression caused by the reward design.
- Failed run kept: `failed_grpo_smoke_prompt_truncation`. With max_prompt_length 768 every reward was 0 (loss and
  gradient exactly 0), because prompts were truncated and 6–25% of completions were clipped. A direct sampling check
  of the policy showed correct answers, which isolated the cause.

## DPO (`dpo_verifier`)
607 pairs from sampling the SFT policy 4× at temperature 1.0 on 1,500 train tasks. Chosen = a verifier-correct
sample, or the solver demonstration on 92 tasks with no correct sample; rejected = a verifier-incorrect sample.
893 tasks with all samples correct produced no pair. `label_source = verifier_v1` (synthetic). β 0.1, lr 5e-6,
1 epoch, new LoRA, 320 s.

- Training-pair accuracy reached 0.74, but loss fell only from 0.694 to 0.664 (margin 0.06): a small update.
- Held out, it changed 3 of 500 decisions (Δ −0.002). This is under-training, not evidence that DPO can't help.
  The learning rate and epochs weren't tuned, because no dev task set was held out for that.

## DPO reruns (`dpo_verifier_v2`, `dpo_verifier_v3_lr5e-5_ep3`)
The first DPO run (`dpo_verifier`) used `max_length=1024` on pairs of up to 1,323 tokens, so TRL **truncated** them. That
run is kept but compromised. `align.py dpo` now refuses truncation (`max_length` 1,600).

| DPO run | lr / epochs | Final train margin, pair accuracy | Correct raw (Δ vs SFT, 95% CI) | Correct wrapped (Δ) | Correct abstention, infeasible | Over-budget errors |
|---|---|---|---|---|---|---|
| v1 (truncated) | 5e-6 / 1 | 0.06, 0.74 | 0.688 (−0.002) | 0.746 | 0.804 | 103 |
| v2 (no truncation) | 5e-6 / 1 | 0.05, 0.68 | 0.686 (−0.004 [−0.014, +0.006]) | 0.750 (+0.002) | 0.804 | 104 |
| v3 (stronger) | 5e-5 / 3 | **8.54, 0.99** | **0.628 (−0.062 [−0.092, −0.032])** | 0.712 (−0.036 [−0.066, −0.008]) | **0.649** | **124** |

- Truncation wasn't why v1 had no effect. At lr 5e-6 for 1 epoch the update is simply too small.
- **When DPO does move the policy (v3), it hurts.** It fits the training pairs almost perfectly but loses 6 points of held-out
  correctness: it abstains far less (false abstention 1%, but correct abstention also drops 0.80 → 0.65) and offers more
  over-budget bundles. The pairs pit an incorrect sample against a correct one, and in many pairs the correct answer
  is a bundle, so pushing away from "rejected" pushed toward offering bundles in general. This is over-optimization on a
  narrow synthetic preference signal. No DPO setting here improved held-out correctness.

## Round 2: GRPO with a graded verifiable reward (`grpo_graded`)
Predeclared step reward: 0 unless strictly parsed and verifier-correct; 1.0 for a correct abstention or the
solver-optimal bundle; 0.6 for a valid but suboptimal bundle. 8 samples per prompt, temperature 1.2, 200 steps,
accumulation 4, same β, lr, and lengths as round 1. 4,858 s, 21.9 GiB peak reserved.

| | SFT (merged) | Round 1, binary | **Round 2, graded** |
|---|---|---|---|
| Correct, raw (Δ vs SFT, 95% CI) | 0.690 | 0.702 (+0.012 [−0.022, +0.044]) | **0.704 (+0.014 [−0.004, +0.030])** |
| Correct, wrapped | 0.748 | 0.754 | 0.754 (+0.006 [−0.010, +0.020]) |
| Optimal among valid bundles, raw | 0.831 | 0.593 | **0.830** |
| Correct abstention, infeasible, raw | 0.804 | 0.856 | 0.835 |
| Over-budget errors, raw | 104 | 83 | 95 |
| Groups with all-equal rewards | — | 88% (1,061/1,200) | **48% (380/800)** |

- The graded reward **removes the optimality regression** (0.830 vs 0.593), confirming that round 1's drop was
  reward hacking caused by the binary reward.
- More exploration halved the uninformative groups, but training reward didn't rise (0.78 → 0.75 over the run) and KL
  stayed around 0.001: the policy moved very little. The held-out correctness change is again **not significant**
  (13 tasks only GRPO got right, 6 only SFT got right).
- Conclusion: on these tasks, the SFT policy is near a plateau that the verifier signal can't push with this budget.
  The remaining errors (arithmetic on totals and budgets) are better handled by the verifier/solver wrapper than by RL.

## What would make the next round informative
1. Done in round 2 (graded reward): optimality regression removed, correctness change still not significant.
2. More exploration (temperature 1.2, 8 samples) to cut the 88% all-equal groups, plus a held-out dev task split for tuning.
3. DPO: tried with stronger settings (v3) and it overfits and hurts. A useful signal needs better pairs (best valid vs worse valid, or human labels) and a dev task split for early stopping.
4. Human preference labels to make RLHF claimable: reward model, then online GRPO against it.
