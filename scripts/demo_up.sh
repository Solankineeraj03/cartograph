#!/usr/bin/env bash
# Start the live Cartograph demo backend on the GPU host, in the foreground. From your laptop:
#   ssh -L 8012:127.0.0.1:8012 <gpu-host> 'bash ~/cartograph/repo/scripts/demo_up.sh'
# then open the demo page and keep the backend URL at http://localhost:8012. Ctrl-C (or closing ssh) stops everything.
# Uses ONE GPU (default 1; GPU 0 is excluded after two Xid 79 events) and refuses to start if any GPU process exists.
set -euo pipefail
GPU=${GPU:-1}
cd ~/cartograph/repo
export HF_HOME=~/cartograph/hf CUDA_VISIBLE_DEVICES=$GPU RAY_TMPDIR=$HOME/cartograph/ray_tmp_demo
APP=~/cartograph/venvs/serve_app/bin; REDIS=~/cartograph/opt/redis-7.4.6/src
W=~/cartograph/work/gate_c_v5; EXPORT=$W/exports/ingest_2; PORT=${PORT:-8012}
LOG=~/cartograph/runs/demo; mkdir -p $LOG
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "A GPU is in use; not starting. Current GPU processes:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; exit 2; fi
cleanup() {
  echo; echo "Stopping demo..."
  [ -n "${SERVE:-}" ] && kill $SERVE 2>/dev/null || true
  $APP/ray stop --force > /dev/null 2>&1 || true
  for p in $(ps -u $USER -o pid=,cmd= | grep "VLLM::EngineCor[e]" | awk '{print $1}'); do kill $p 2>/dev/null || true; done
  $REDIS/redis-cli -p 6390 shutdown nosave > /dev/null 2>&1 || true
  echo "Stopped. GPU processes left: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
}
trap cleanup EXIT INT TERM HUP
echo "[1/3] Redis + feature materialization (snapshot ingest_2, watermark 2026-10-01T02:00Z)..."
$REDIS/redis-server --port 6390 --bind 127.0.0.1 --save "" --appendonly no --daemonize yes > /dev/null
$APP/python -m cartograph.materialize --repo $W/feature_repo --export $EXPORT --end 2026-10-01T02:00:00+00:00 > $LOG/materialize.log 2>&1
echo "[2/3] Starting Ray Serve + vLLM on GPU $GPU (about 1-2 minutes)..."
CARTOGRAPH_MODEL=~/cartograph/checkpoints/align/sft_merged CARTOGRAPH_RELEASE=2406781e60ea \
CARTOGRAPH_SLICE=~/cartograph/data/processed/esci_us_small_v1 CARTOGRAPH_SIM=~/cartograph/data/ops_sim_v1 \
CARTOGRAPH_EXPORT=$EXPORT CARTOGRAPH_FEATURE_REPO=$W/feature_repo CARTOGRAPH_PORT=$PORT \
$APP/python -m cartograph.serve_app > $LOG/serve.log 2>&1 &
SERVE=$!
for _ in $(seq 240); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break
  kill -0 $SERVE 2>/dev/null || { echo "serve exited; see $LOG/serve.log"; tail -20 $LOG/serve.log; exit 1; }; sleep 2; done
curl -sf http://127.0.0.1:$PORT/v1/info > /dev/null || { echo "serve did not become ready"; exit 1; }
echo "[3/3] Ready: http://localhost:$PORT (through your ssh -L tunnel). Press Ctrl-C to stop."
wait $SERVE
