#!/usr/bin/env bash
# Ray Serve request path on ONE GPU (default 1): live Redis features, in-process vLLM, verify/retry, traces.
# Then: Ray Serve vs bare vLLM (same prompts, vllm bench), 500 served task requests, stale/outage probes, replica failover.
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
$APP/python -m cartograph.materialize --repo $W/feature_repo --export $EXPORT --end 2026-10-01T02:00:00+00:00 > $R/materialize.log 2>&1
echo "done materialize"
(while true; do echo "$(date +%s.%N),$(nvidia-smi -i $GPU --query-gpu=memory.used,power.draw,utilization.gpu --format=csv,noheader,nounits)"; sleep 0.5; done) > $R/gpu_samples.csv &
POLL=$!
CARTOGRAPH_MODEL=$POLICY CARTOGRAPH_RELEASE=2406781e60ea CARTOGRAPH_SLICE=~/cartograph/data/processed/esci_us_small_v1 \
CARTOGRAPH_SIM=~/cartograph/data/ops_sim_v1 CARTOGRAPH_EXPORT=$EXPORT CARTOGRAPH_FEATURE_REPO=$W/feature_repo \
CARTOGRAPH_PORT=$PORT $APP/python -m cartograph.serve_app > $R/serve.log 2>&1 &
SERVE=$!
for _ in $(seq 240); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 2; done
curl -sf http://127.0.0.1:$PORT/health || { echo "serve failed to start"; tail -30 $R/serve.log; kill $SERVE $POLL; exit 1; }
echo "done serve_start"
for c in 1 8 32; do n=$(( c < 25 ? 100 : 128 ))
  $SV/vllm bench serve --backend vllm --base-url http://127.0.0.1:$PORT --model cartograph --tokenizer $POLICY \
    --dataset-name custom --dataset-path $P/prompts.jsonl --custom-skip-chat-template --custom-output-len 128 --ignore-eos \
    --num-prompts $n --max-concurrency $c --request-rate inf --temperature 0 --seed 13 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir $R --result-filename bench_c$c.json > $R/bench_c$c.log 2>&1 || echo "bench c$c failed"
  echo "done bench_c$c"; done
$APP/python -m cartograph.serve_eval tasks --tasks ~/cartograph/data/tasks_v1/tasks_test.jsonl --now 2026-10-01T04:00:00+00:00 --out $R/tasks > $R/tasks.log 2>&1 || echo "tasks failed"
echo "done tasks"
$APP/python -m cartograph.serve_eval probe --request "Find a quiet wireless keyboard and mouse bundle under \$100. Only include available products and explain the tradeoffs." --now 2026-10-01T04:00:00+00:00 --out $R/probe_fresh.json || true
$APP/python -m cartograph.serve_eval probe --request "Find a quiet wireless keyboard and mouse bundle under \$100." --now 2026-10-01T09:30:00+00:00 --out $R/probe_stale.json || true
$APP/python -m cartograph.serve_eval failover --tasks ~/cartograph/data/tasks_v1/tasks_test.jsonl --now 2026-10-01T04:00:00+00:00 --requests 40 --kill-after 10 --out $R/failover > $R/failover.log 2>&1 || echo "failover failed"
echo "done failover"
$REDIS/redis-cli -p 6390 shutdown nosave || true
sleep 1
$APP/python -m cartograph.serve_eval probe --request "Find a quiet wireless keyboard and mouse bundle under \$100." --now 2026-10-01T04:00:00+00:00 --out $R/probe_redis_down.json || true
echo "done probes"
kill $SERVE; wait $SERVE 2>/dev/null || true; kill $POLL
$APP/ray stop --force > /dev/null 2>&1 || true
# vLLM's engine subprocess can outlive Ray; kill our own leftovers so the GPU is really free.
for p in $(ps -u $USER -o pid=,cmd= | grep "VLLM::EngineCor[e]" | awk '{print $1}'); do kill $p; done
echo ALL_DONE
