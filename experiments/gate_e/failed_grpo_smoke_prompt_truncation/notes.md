# Failed run: GRPO smoke test 1

All 48 completions got reward 0 (12/12 groups all-equal), so loss and grad_norm were 0. The same merged SFT policy scores
correct answers when sampled directly (debug script, temperature 1.0). Causes: max_prompt_length=768 truncated ~950-token
prompts, and max_completion_length=256 clipped 6-25% of completions mid-JSON. Fix: 1280/384 plus a guard that refuses
to train on truncated prompts.
