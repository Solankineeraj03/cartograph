#!/usr/bin/env bash
# Gate E on ONE GPU (default GPU 1). All branches start from the same merged SFT checkpoint.
# RLVR: GRPO with the deterministic verifier as reward. DPO: verifier-labeled pairs (synthetic, not human feedback).
set -euo pipefail
GPU=${GPU:-1}
cd ~/cartograph/repo
export HF_HOME=~/cartograph/hf CUDA_VISIBLE_DEVICES=$GPU
T=~/cartograph/data/tasks_v1; R=~/cartograph/runs/gate_e; C=~/cartograph/checkpoints/align
TR=~/cartograph/venvs/training/bin/python; SV=~/cartograph/venvs/serving/bin/python
POLICY=$C/sft_merged
mkdir -p $R $C
# Busy if any GPU process persists for 60 s (a just-finished vLLM engine can take a few seconds to exit).
busy() { for _ in $(seq 12); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 1; sleep 5; done; return 0; }
step() { local name=$1; shift; [ -f $R/$name.done ] && { echo "skip $name"; return; }
  if busy; then echo "GPU busy before $name; stopping"; exit 2; fi
  if "$@" > $R/$name.log 2>&1; then touch $R/$name.done; [ -d $R/$name ] && cp $R/$name.log $R/$name/raw.log; echo "done $name";
  else echo "FAILED $name"; tail -3 $R/$name.log; exit 1; fi; }
step merge $TR -m cartograph.align merge --adapter ~/cartograph/checkpoints/sft/full_v1/adapter --out $POLICY
step grpo_smoke $TR -m cartograph.align grpo --policy $POLICY --tasks $T/tasks_train.jsonl --out $C/grpo_smoke --evidence $R/grpo_smoke --steps 3 --max-tasks 64
step grpo_rlvr $TR -m cartograph.align grpo --policy $POLICY --tasks $T/tasks_train.jsonl --out $C/grpo_rlvr --evidence $R/grpo_rlvr --steps 300
step grpo_rlvr_test $SV -m cartograph.eval_assistant --tasks $T/tasks_test.jsonl --engine vllm --model $POLICY --adapter $C/grpo_rlvr/adapter --evidence $R/grpo_rlvr_test
step sft_merged_test $SV -m cartograph.eval_assistant --tasks $T/tasks_test.jsonl --engine vllm --model $POLICY --evidence $R/sft_merged_test
step pairs $SV -m cartograph.align pairs --policy $POLICY --tasks $T/tasks_train.jsonl --out $R/pairs_verifier_v1.jsonl --max-tasks 1500 --samples 4
step dpo $TR -m cartograph.align dpo --policy $POLICY --pairs $R/pairs_verifier_v1.jsonl --out $C/dpo_verifier --evidence $R/dpo_verifier
step dpo_test $SV -m cartograph.eval_assistant --tasks $T/tasks_test.jsonl --engine vllm --model $POLICY --adapter $C/dpo_verifier/adapter --evidence $R/dpo_verifier_test
# Round 2: graded verifiable reward, more exploration (8 samples, temperature 1.2).
step grpo_graded $TR -m cartograph.align grpo --policy $POLICY --tasks $T/tasks_train.jsonl --out $C/grpo_graded --evidence $R/grpo_graded --reward graded --group 8 --accum 4 --steps 200 --temperature 1.2
step grpo_graded_test $SV -m cartograph.eval_assistant --tasks $T/tasks_test.jsonl --engine vllm --model $POLICY --adapter $C/grpo_graded/adapter --evidence $R/grpo_graded_test
echo ALL_DONE
