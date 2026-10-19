# Pipeline smoke test: reward model → GRPO against it (synthetic labels, NOT RLHF)

Purpose: prove the RLHF code path (`scripts/rlhf.sh`) runs before any human labels are spent on it. Labels here are the
verifier pairs (`label_source = verifier_v1`); GRPO needed `--allow-synthetic-rm`. **None of this is human-feedback evidence.**

- First attempt (`failed_rm_truncated_1step`): the reward model trained for **1 step on about 5 examples**, with held-out
  pairwise accuracy **0.15**. TRL's RewardTrainer silently dropped pairs longer than `max_length=1024` (prompt plus answer
  is about 1.2–1.3k tokens), and the scorer truncated the answer away. The GRPO run on it is kept as `grpo_on_failed_rm`.
- Fix: `max_length` 1,600, a guard that refuses to train if any sequence would be dropped or truncated, scoring without
  truncation, and the number of training steps recorded.
- Second attempt (`rm`, `grpo`): 32 steps over all 500 pairs (longest sequence 1,323 tokens), held-out pairwise
  accuracy **0.617** on 107 pairs. A 3-step GRPO run against this reward model finished.
- The same review found the earlier DPO run used `max_length=1024` on the same-length pairs, so it was truncated.
  It has been rerun (`dpo_verifier_v2`).
