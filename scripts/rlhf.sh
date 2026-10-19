#!/usr/bin/env bash
# Human-feedback RL, after the owner has labeled pairs (data/human_prefs_v1.jsonl, label_source human_owner_v1).
# split labels -> reward model (Qwen2.5-0.5B + LoRA, scalar head) with held-out pairwise accuracy -> online GRPO
# against that reward, branched from the same merged SFT checkpoint -> held-out task evaluation. One GPU (default 1).
set -euo pipefail
GPU=${GPU:-1}
cd ~/cartograph/repo
export HF_HOME=~/cartograph/hf CUDA_VISIBLE_DEVICES=$GPU
TR=~/cartograph/venvs/training/bin/python; SV=~/cartograph/venvs/serving/bin/python
R=~/cartograph/runs/gate_e; C=~/cartograph/checkpoints/align; POLICY=$C/sft_merged
LABELS=${LABELS:-~/cartograph/data/human_prefs_v1.jsonl}
test -f "$LABELS" || { echo "no labels at $LABELS"; exit 1; }
$TR -m cartograph.align split_labels --labels $LABELS --train-out $R/human_pairs_train.jsonl --test-out $R/human_pairs_test.jsonl | tee $R/human_split.json
$TR -m cartograph.align rm --pairs $R/human_pairs_train.jsonl --test-pairs $R/human_pairs_test.jsonl --out $C/rm_human_v1 --evidence $R/rm_human_v1 > $R/rm_human_v1.log 2>&1
echo "done rm"
$TR -m cartograph.align grpo --policy $POLICY --tasks ~/cartograph/data/tasks_v1/tasks_train.jsonl --reward $C/rm_human_v1/reward_model \
  --out $C/grpo_rlhf_v1 --evidence $R/grpo_rlhf_v1 --group 4 --steps 200 > $R/grpo_rlhf_v1.log 2>&1
echo "done grpo_rlhf"
$SV -m cartograph.eval_assistant --tasks ~/cartograph/data/tasks_v1/tasks_test.jsonl --engine vllm --model $POLICY \
  --adapter $C/grpo_rlhf_v1/adapter --evidence $R/grpo_rlhf_v1_test > $R/grpo_rlhf_v1_test.log 2>&1
echo ALL_DONE
