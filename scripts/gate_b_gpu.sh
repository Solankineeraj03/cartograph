#!/usr/bin/env bash
# Gate B variants on one GPU: scripts/gate_b_gpu.sh <gpu> <variant>...
# Same budget as the CPU chain (1000 steps, 32 queries, seed 13). Training uses bf16 autocast on GPU;
# encoding uses fp32 so vectors match the CPU-encoded variants. Exact evaluation runs on CPU.
set -euo pipefail
GPU=$1; shift
UUID=$(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader)
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$UUID"; then echo "GPU $GPU busy"; exit 2; fi
cd ~/cartograph/repo
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=~/cartograph/hf OMP_NUM_THREADS=12
PY="$HOME/cartograph/venvs/retrieval/bin/python"
SLICE=~/cartograph/data/processed/esci_us_small_v1
R=~/cartograph/runs/gate_b; C=~/cartograph/checkpoints/encoders; V=~/cartograph/data/vectors
MINILM="--base sentence-transformers/all-MiniLM-L6-v2 --base-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --lr 2e-5"
RAW="--base nreimers/MiniLM-L6-H384-uncased --base-revision 3276f0fac9d818781d7a1327b3ff818fc4e643c0 --lr 5e-5"
COMMON="--steps 1000 --batch-queries 32 --device cuda --seed 13 --slice $SLICE"
declare -A ARGS=(
  [b4_matryoshka_mean]="$MINILM --pooling mean --objective contrastive --matryoshka 384,256,128,64"
  [b5_raw_mean]="$RAW --pooling mean --objective contrastive"
  [b6_raw_cls]="$RAW --pooling cls --objective contrastive")
declare -A DIMS=([b4_matryoshka_mean]="--dims=256,128,64 --prefix-only" [b5_raw_mean]="--dims=" [b6_raw_cls]="--dims=")
for name in "$@"; do
  if [ "$name" = frozen_dev ]; then
    [ -f $R/eval_frozen_dev/manifest.json ] || $PY -m cartograph.dims --slice $SLICE --product-vectors $V/minilm-frozen/products.npy \
      --query-vectors $V/minilm-frozen/queries_dev.npy --split dev --dims= --threads 12 --evidence $R/eval_frozen_dev > $R/eval_frozen_dev.log 2>&1
    echo "done $name"; continue
  fi
  [ -f $R/train_$name/manifest.json ] || $PY -m cartograph.encoders $COMMON ${ARGS[$name]} --out $C/$name --evidence $R/train_$name > $R/train_$name.log 2>&1
  [ -f $V/$name/encode_manifest.json ] || { rm -rf $V/$name; $PY -m cartograph.encode --slice $SLICE --model $C/$name --out $V/$name --device cuda --fp32 --batch-size 256 > $R/encode_$name.log 2>&1; }
  for split in dev test; do
    [ -f $R/eval_${name}_$split/manifest.json ] || $PY -m cartograph.dims --slice $SLICE --product-vectors $V/$name/products.npy \
      --query-vectors $V/$name/queries_$split.npy --split $split ${DIMS[$name]} --threads 12 --evidence $R/eval_${name}_$split > $R/eval_${name}_$split.log 2>&1
  done
  echo "done $name"
done
echo ALL_DONE
