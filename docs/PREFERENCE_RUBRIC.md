# Preference rubric (pref-rubric-v1)

Used by `python -m cartograph.label` for owner-annotated pairwise preferences. These are the only
labels that count as human feedback for the RLHF experiment (`label_source = human_owner_v1`).

Judge each answer only against the request and the catalog evidence shown. In priority order:

1. **Correctness.** One in-stock product per required category, no duplicates or invented IDs,
   a correct total, and within budget. If no valid bundle exists, the better answer says so and
   names the failing constraint.
2. **Grounding.** Every price, stock status, and product claim matches the evidence. Penalize
   claims about features the evidence does not show.
3. **Usefulness.** The explanation states a real tradeoff (price vs. match to the request) in one
   or two sentences.
4. **Brevity.** Between two equally good answers, prefer the shorter one. Never prefer an answer
   only for being longer.

Pick tie when the answers are equally good or equally wrong. Skip when you can't decide from the
evidence. Don't look at the verifier's result or the shown order before deciding.
Target: 300–500 labeled pairs from train-split tasks only. Hold out 20% by task_id for evaluating the reward model.

## How to label (owner)
Candidates: `experiments/gate_e/human_candidates_v1.jsonl`. 500 pairs sampled from the SFT policy on train tasks:
about 80% are **both-valid** pairs (the verifier accepts both, so only your judgment can rank them) and about 20% are
**controls** (one valid, one invalid), used to measure agreement with the verifier. The pair type is hidden while you
label and recorded afterwards.
```
python3 -m cartograph.label --candidates experiments/gate_e/human_candidates_v1.jsonl \
  --out data/human_prefs_v1.jsonl --annotator <your name>
```
It's resumable: quit with `q` and rerun to continue. Aim for at least 300 labels. Then the next steps are a reward model
on these labels (`align.py rm`, 80/20 split by task) and online GRPO against it (`align.py grpo --reward <rm dir>`), which
refuses to run unless the reward model's `label_source` is `human_owner_v1`.
