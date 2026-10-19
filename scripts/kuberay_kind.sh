#!/usr/bin/env bash
# KubeRay drill on a local kind cluster (macOS Docker Desktop, CPU only). Evidence: experiments/gate_g/kuberay_kind/.
# Deploy v1 -> cross-environment replay -> delete the worker pod under traffic -> upgrade to v2 -> roll back to v1 -> replay.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=experiments/gate_g/kuberay_kind; mkdir -p $OUT
KUBERAY=v1.7.1; V1=2406781e60ea; V2=31504bd1c8ea
IMAGE=cartograph-cpu:$(git rev-parse --short HEAD)
test -z "$(git status --porcelain --untracked-files=no)" || { echo "commit first"; exit 1; }
docker info > /dev/null
kind get clusters | grep -q '^cartograph$' || kind create cluster --name cartograph --wait 120s
kubectl config use-context kind-cartograph
kubectl apply --server-side -k "github.com/ray-project/kuberay/ray-operator/config/default?ref=$KUBERAY" > $OUT/operator_install.log
kubectl -n default set resources deploy/kuberay-operator --requests=cpu=50m,memory=128Mi --limits=cpu=500m,memory=256Mi
kubectl -n default rollout status deploy/kuberay-operator --timeout=300s || kubectl get pods -A
docker build -t $IMAGE -f deploy/kuberay/Dockerfile . > $OUT/docker_build.log 2>&1
kind load docker-image $IMAGE --name cartograph
deploy() { sed -e "s#__RELEASE__#$1#" -e "s#__IMAGE__#$IMAGE#" deploy/kuberay/rayservice.yaml.tmpl | kubectl apply -f -
  for _ in $(seq 120); do
    [ "$(kubectl get rayservice cartograph -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)" = True ] && \
    kubectl get rayservice cartograph -o json | python3 -c "import json,sys;d=json.load(sys.stdin);a=d['status'].get('activeServiceStatus',{});sys.exit(0 if '$1' in json.dumps(d['spec']) else 1)" && break; sleep 5; done
  kubectl get rayservice cartograph -o yaml > $OUT/rayservice_$1.yaml; }
forward() { pkill -f "port-forward svc/cartograph-serve-svc" || true; sleep 1
  kubectl port-forward svc/cartograph-serve-svc 18000:8000 > /dev/null 2>&1 & sleep 3
  for _ in $(seq 60); do curl -sf http://127.0.0.1:18000/health > /dev/null && return 0; sleep 2; done; return 1; }
date -u +%FT%TZ > $OUT/started_at.txt
deploy $V1; forward
python3 -m cartograph.k8s_drill replay --out $OUT/replay_v1.json --reference experiments/gate_g/governance_drill/traces.jsonl
kubectl get pods -o wide > $OUT/pods_before_failure.txt
# Pod failure under traffic: delete the worker pod hosting the Serve replica; one request per second for up to 240 s.
python3 -m cartograph.k8s_drill watch --seconds 240 --out $OUT/failover.json &
WATCH=$!
sleep 10
WORKER=$(kubectl get pods -l ray.io/node-type=worker -o jsonpath='{.items[0].metadata.name}')
date -u +%FT%T.%3NZ > $OUT/worker_deleted_at.txt; echo $WORKER >> $OUT/worker_deleted_at.txt
kubectl delete pod $WORKER --wait=false
sleep 150; touch $OUT/failover.stop; wait $WATCH || true; rm -f $OUT/failover.stop
kubectl get pods -o wide > $OUT/pods_after_failure.txt
forward
# Upgrade to v2, then roll back to v1 and replay.
deploy $V2; forward
python3 -m cartograph.k8s_drill replay --out $OUT/replay_v2.json --reference experiments/gate_g/governance_drill/traces.jsonl
deploy $V1; forward
python3 -m cartograph.k8s_drill replay --out $OUT/replay_after_rollback.json --reference experiments/gate_g/governance_drill/traces.jsonl
kubectl get events --sort-by=.lastTimestamp > $OUT/events.txt
kubectl version -o json > $OUT/kubectl_version.json; kind version > $OUT/kind_version.txt; docker version --format '{{json .}}' > $OUT/docker_version.json
docker image inspect $IMAGE --format '{{json .Id}}' > $OUT/image_id.txt
pkill -f "port-forward svc/cartograph-serve-svc" || true
echo "done; delete with: kind delete cluster --name cartograph"
