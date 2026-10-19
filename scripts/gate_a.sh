#!/usr/bin/env bash
# Gate A on the GPU host. Run from the Mac:  scripts/gate_a.sh
# Syncs the committed tree (with .git, so manifests record the commit), builds the audited
# ESCI slice, runs BM25 / untouched MiniLM / RRF on the official test split, and pulls evidence back.
set -euo pipefail
HOST=${HOST:-gpu-host}
ESCI=7916cdf6ab75a462e77f20ab40428a10923998d5
cd "$(dirname "$0")/.."
test -z "$(git status --porcelain --untracked-files=no)" || { echo "commit first"; exit 1; }
rsync -az --exclude artifacts --exclude __pycache__ --exclude experiments ./ "$HOST":cartograph/repo/
ssh "$HOST" bash -s <<REMOTE
set -euo pipefail
export HF_HOME=\$HOME/cartograph/hf PATH=\$HOME/.local/bin:\$PATH
cd ~/cartograph/repo
PY=~/cartograph/venvs/retrieval/bin/python
GPU=\${GPU:-0}
UUID=\$(nvidia-smi -i \$GPU --query-gpu=uuid --format=csv,noheader)
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "\$UUID"; then
  echo "GPU \$GPU busy with other processes; backing off"; nvidia-smi; exit 2; fi
RUNS=~/cartograph/runs/gate_a
SLICE=~/cartograph/data/processed/esci_us_small_v1
mkdir -p \$RUNS
\$PY -m cartograph.esci --raw ~/cartograph/data/raw/esci/$ESCI --out \$SLICE \
  --evidence \$RUNS/esci_slice 2>&1 | tee \$RUNS/esci_slice.log
CUDA_VISIBLE_DEVICES=\$GPU \$PY -m cartograph.eval_retrieval --slice \$SLICE --split test \
  --evidence \$RUNS/baselines_test \
  --embedding-cache ~/cartograph/data/embeddings/minilm-1110a243/\$(python3 -c "import json;print(json.load(open('\$SLICE/data_manifest.json'))['slice_id'])").npy \
  2>&1 | tee \$RUNS/baselines_test.log
REMOTE
mkdir -p experiments/gate_a
rsync -az "$HOST":cartograph/runs/gate_a/ experiments/gate_a/
rsync -az "$HOST":cartograph/repo/envs/retrieval.lock.txt envs/
echo "Evidence in experiments/gate_a"
