#!/usr/bin/env bash
# Ray Serve request path on ONE GPU (default 1): live Redis features, in-process vLLM, verify/retry, traces.
# Rerun after the request-parser fix: 500 served task requests and the replica-failover drill (benchmarks and probes not repeated).
set -euo pipefail
GPU=${GPU:-1}
cd ~/cartograph/repo
export HF_HOME=~/cartograph/hf CUDA_VISIBLE_DEVICES=$GPU RAY_TMPDIR=$HOME/cartograph/ray_tmp_serve
APP=~/cartograph/venvs/serve_app/bin; SV=~/cartograph/venvs/serving/bin
POLICY=~/cartograph/checkpoints/align/sft_merged
R=~/cartograph/runs/gate_f/ray_serve; P=~/cartograph/runs/gate_f/prompts
W=~/cartograph/work/gate_c_v5; EXPORT=$W/exports/ingest_2
REDIS=~/cartograph/opt/redis-7.4.6/src; PORT=8012
mkdir -p $R
for _ in $(seq 12); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && { echo "GPU busy"; exit 2; }
$REDIS/redis-server --port 6390 --bind 127.0.0.1 --save "" --appendonly no --daemonize yes
sleep 1
$APP/python -m cartograph.materialize --repo $W/feature_repo --export $EXPORT --end 2026-10-01T02:00:00+00:00 > $R/materialize_v2.log 2>&1
echo "done materialize"
(while true; do echo "$(date +%s.%N),$(nvidia-smi -i $GPU --query-gpu=memory.used,power.draw,utilization.gpu --format=csv,noheader,nounits)"; sleep 0.5; done) > $R/gpu_samples_v2.csv &
POLL=$!
CARTOGRAPH_MODEL=$POLICY CARTOGRAPH_RELEASE=2406781e60ea CARTOGRAPH_SLICE=~/cartograph/data/processed/esci_us_small_v1 \
CARTOGRAPH_SIM=~/cartograph/data/ops_sim_v1 CARTOGRAPH_EXPORT=$EXPORT CARTOGRAPH_FEATURE_REPO=$W/feature_repo \
CARTOGRAPH_PORT=$PORT $APP/python -m cartograph.serve_app > $R/serve_v2.log 2>&1 &
SERVE=$!
for _ in $(seq 240); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 2; done
curl -sf http://127.0.0.1:$PORT/health || { echo "serve failed to start"; tail -30 $R/serve_v2.log; kill $SERVE $POLL; exit 1; }
echo "done serve_start"
$APP/python -m cartograph.serve_eval tasks --tasks ~/cartograph/data/tasks_v1/tasks_test.jsonl --now 2026-10-01T04:00:00+00:00 --out $R/tasks_v2_parser_fixed > $R/tasks_v2_parser_fixed.log 2>&1 || echo "tasks failed"
echo "done tasks_v2"
$APP/python -m cartograph.serve_eval failover --tasks ~/cartograph/data/tasks_v1/tasks_test.jsonl --now 2026-10-01T04:00:00+00:00 --requests 60 --kill-after 10 --out $R/failover_v2 > $R/failover_v2.log 2>&1 || echo "failover failed"
echo "done failover_v2"
$REDIS/redis-cli -p 6390 shutdown nosave || true
kill $SERVE; wait $SERVE 2>/dev/null || true; kill $POLL
$APP/ray stop --force > /dev/null 2>&1 || true
# vLLM's engine subprocess can outlive Ray; kill our own leftovers so the GPU is really free.
for p in $(ps -u $USER -o pid=,cmd= | grep "VLLM::EngineCor[e]" | awk '{print $1}'); do kill $p; done
echo ALL_DONE
