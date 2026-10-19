#!/usr/bin/env bash
# Gate D on ONE GPU (default GPU 1; GPU 0 is excluded after two Xid 79 events). Steps run sequentially.
set -euo pipefail
GPU=${GPU:-1}
cd ~/cartograph/repo
export HF_HOME=~/cartograph/hf
T=~/cartograph/data/tasks_v1; R=~/cartograph/runs/gate_d; C=~/cartograph/checkpoints/sft
mkdir -p $R
# Busy if any GPU process persists for 60 s (a just-finished vLLM engine can take a few seconds to exit).
busy() { for _ in $(seq 12); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 1; sleep 5; done; return 0; }
step() { local name=$1; shift; [ -f $R/$name/manifest.json ] && { echo "skip $name"; return; }
  if busy; then echo "GPU busy before $name; stopping"; nvidia-smi; exit 2; fi
  "$@" > $R/$name.log 2>&1 || { echo "FAILED $name"; tail -5 $R/$name.log; }
  [ -d $R/$name ] && cp $R/$name.log $R/$name/raw.log; echo "done $name"; }
cp $T/tasks_manifest.json $R/tasks_manifest.json
step solver_test env CUDA_VISIBLE_DEVICES= ~/cartograph/venvs/retrieval/bin/python -m cartograph.eval_assistant --tasks $T/tasks_test.jsonl --engine solver --evidence $R/solver_test
step base_vllm_test env CUDA_VISIBLE_DEVICES=$GPU ~/cartograph/venvs/serving/bin/python -m cartograph.eval_assistant --tasks $T/tasks_test.jsonl --engine vllm --model qwen2.5-1.5b-instruct --evidence $R/base_vllm_test
step sft_pilot env CUDA_VISIBLE_DEVICES=$GPU ~/cartograph/venvs/training/bin/python -m cartograph.sft --tasks $T/tasks_train.jsonl --out $C/pilot --evidence $R/sft_pilot --pilot-steps 20 --max-length 1536
GC=""
if ! python3 -c "import json,sys;m=json.load(open('$R/sft_pilot/metrics.json'));sys.exit(0 if m['peak_reserved_gib']<20 else 1)"; then GC="--gradient-checkpointing"; fi
echo "full SFT gradient checkpointing: ${GC:-off}"
step sft_full env CUDA_VISIBLE_DEVICES=$GPU ~/cartograph/venvs/training/bin/python -m cartograph.sft --tasks $T/tasks_train.jsonl --out $C/full_v1 --evidence $R/sft_full --max-length 1536 $GC
step sft_vllm_test env CUDA_VISIBLE_DEVICES=$GPU ~/cartograph/venvs/serving/bin/python -m cartograph.eval_assistant --tasks $T/tasks_test.jsonl --engine vllm --model qwen2.5-1.5b-instruct --adapter $C/full_v1/adapter --evidence $R/sft_vllm_test
echo ALL_DONE
