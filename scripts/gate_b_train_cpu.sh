#!/usr/bin/env bash
# Gate B encoder variants on CPU (GPU host unavailable after Xid 79). Identical budget for every variant:
# 1000 steps, 32 queries/step (+32 query-aware hard negatives), tau 0.05, seed 13, doc 256 / query 32 tokens.
# Each variant: train -> encode catalog + dev/test queries -> exact full-catalog eval on dev and on test.
set -euo pipefail
cd ~/cartograph/repo
export CUDA_VISIBLE_DEVICES= HF_HOME=~/cartograph/hf OMP_NUM_THREADS=24
PY="nice -n 10 $HOME/cartograph/venvs/retrieval/bin/python"
SLICE=~/cartograph/data/processed/esci_us_small_v1
R=~/cartograph/runs/gate_b; C=~/cartograph/checkpoints/encoders; V=~/cartograph/data/vectors
MINILM="--base sentence-transformers/all-MiniLM-L6-v2 --base-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --lr 2e-5"
RAW="--base nreimers/MiniLM-L6-H384-uncased --base-revision 3276f0fac9d818781d7a1327b3ff818fc4e643c0 --lr 5e-5"
COMMON="--steps 1000 --batch-queries 32 --device cpu --threads 24 --seed 13 --slice $SLICE"
run() {  # name, eval-dims-args, trainer args...
  local name=$1 dims=$2; shift 2
  [ -f $R/train_$name/manifest.json ] || $PY -m cartograph.encoders $COMMON "$@" --out $C/$name --evidence $R/train_$name > $R/train_$name.log 2>&1
  [ -f $V/$name/encode_manifest.json ] || $PY -m cartograph.encode --slice $SLICE --model $C/$name --out $V/$name --device cpu --threads 24 > $R/encode_$name.log 2>&1
  for split in dev test; do
    [ -f $R/eval_${name}_$split/manifest.json ] || $PY -m cartograph.dims --slice $SLICE --product-vectors $V/$name/products.npy \
      --query-vectors $V/$name/queries_$split.npy --split $split $dims --threads 24 --evidence $R/eval_${name}_$split > $R/eval_${name}_$split.log 2>&1
  done
  echo "done $name"
}
run b1_contrastive_mean --dims= $MINILM --pooling mean --objective contrastive
run b3_combined_mean --dims= $MINILM --pooling mean --objective contrastive+graded
run b2_graded_mean --dims= $MINILM --pooling mean --objective graded
run b4_matryoshka_mean "--dims=256,128,64 --prefix-only" $MINILM --pooling mean --objective contrastive --matryoshka 384,256,128,64
run b5_raw_mean --dims= $RAW --pooling mean --objective contrastive
run b6_raw_cls --dims= $RAW --pooling cls --objective contrastive
# Frozen encoder on dev, for selection against the same rule.
[ -f $R/eval_frozen_dev/manifest.json ] || $PY -m cartograph.dims --slice $SLICE --product-vectors $V/minilm-frozen/products.npy \
  --query-vectors $V/minilm-frozen/queries_dev.npy --split dev --dims= --threads 24 --evidence $R/eval_frozen_dev > $R/eval_frozen_dev.log 2>&1
echo ALL_DONE
