# Bounded cloud experiment: plan (not executed)

Status: `planned`. No cloud resources have been created. The owner has no cloud account available for this
project yet (2026-10-18). Everything below is prepared so the run can start as soon as an account and spending cap
exist. Prices must be read from the provider's pricing page on the day of the run; this file deliberately doesn't
quote a rate.

## What it has to prove (and what the lab host can't)

1. **Tensor parallelism:** TP=1 vs TP=2 for the same model on the same host and GPU type, plus two independent
   TP=1 replicas when the model fits on one GPU. The lab host has two RTX 3090s, but GPU 0 has fallen off the bus
   twice under dual-GPU load (Xid 79), so no two-GPU measurement there is safe or trustworthy.
2. **GPU scheduling on Kubernetes:** `deploy/kuberay/rayservice-gpu.yaml` with the NVIDIA device plugin, where the
   local kind drill can only show CPU scheduling.

## Resource choice

One on-demand instance with **≥ 2 identical NVIDIA GPUs and ≥ 24 GB each**, in one region, for a single session:
- AWS: a g5 or g6 family instance with 4 GPUs (A10G or L4, 24 GB each). Use 2 of its GPUs for TP=2 and 1 for TP=1,
  so both use the same GPU type, driver, and interconnect.
- Record `nvidia-smi topo -m` (PCIe vs NVLink): TP's communication cost depends on it.

## Exact procedure (target ≤ 3 instance-hours)

1. Launch the instance from a current Deep Learning AMI (record the AMI ID). Build `envs/serving.lock.txt`
   (vLLM 0.11.0) and copy `sft_merged` (about 3 GB, SHA-256 in the Gate E evidence) and `runs/gate_f/prompts`.
2. Run `scripts/gate_f.sh` subsets, unchanged, with:
   - `--tensor-parallel-size 1` on GPU 0; concurrency 1, 8, 32, 64.
   - `--tensor-parallel-size 2` on GPUs 0,1; same sweep.
   - Two TP=1 servers on GPUs 0 and 1 behind a round-robin split of the same prompts.
   Same prompts, output length 128, `--ignore-eos`, greedy decoding.
3. Optional (≤ 1 h): k3s plus the NVIDIA device plugin plus KubeRay v1.7.1; apply `rayservice-gpu.yaml`; delete the
   GPU worker pod under traffic; record recovery.
4. Collect `nvidia-smi` samples, vLLM logs (communication setup), bench JSONs, and run manifests into
   `experiments/gate_g/cloud_tp/`.

## Cost estimate formula (fill in on the day)

`cost = instance_hours × on_demand_rate + EBS_GB × gp3_rate × hours/730 + egress_GB × egress_rate`
Expect about 3 instance-hours, a 200 GB gp3 volume, and under 5 GB of egress (results only).
Cost per successful request at a latency bound = `instance_rate_per_second × wall_seconds / successful_requests`,
reported per configuration.

## Cleanup (mandatory, same session)

Terminate the instance, delete the EBS volume and any snapshots or AMIs created, delete the security group and
key pair, and confirm in the billing console that no resources remain. Tag every resource `project=cartograph`
and list them with `aws resourcegroupstaggingapi get-resources --tag-filters Key=project,Values=cartograph`
before and after.
