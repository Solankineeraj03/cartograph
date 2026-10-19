# Failed run: second GPU 0 "fallen off the bus" (2026-10-15)

- Context: the host had been rebooted after the first incident (2026-10-14, `experiments/gate_a/failed_baselines_test_gpu0_xid79`).
  Both GPUs then passed a 50-iteration bf16 8192² matmul check (58.7 and 62.1 TFLOPS, finite results, no Xid).
- Workload: GPU 0 ran `cartograph.encode` (fine-tuned MiniLM b4, fp32, batch 256, the 482k-product catalog), the same kind
  of sustained encoding job that triggered the first incident. GPU 1 had just finished training b6 (1,000 steps,
  71 s, 3.47 GiB peak reserved; `train_b6_raw_cls/` is a valid, completed run) and had started encoding.
- Event: at uptime 33,028 s the kernel logged `Xid 79: GPU has fallen off the bus` for PCI 0000:01:00 (GPU 0), then
  `Xid 154 ... Node Reboot Required` for both GPUs. Both encode jobs died with CUDA launch failures.
- Assessment: two Xid 79 events on the same card under similar moderate sustained load (about 4 GB, roughly 1 minute
  in), plus earlier Xid 31/43 faults on that card under other users' jobs. This points to a hardware, power, or PCIe
  problem with GPU 0 (riser, seating, or PSU rail), not something the software can work around. Root access is needed
  for `nvidia-bug-report.sh`.
- Action: all GPU work on this host is stopped. GPU 0 should not be used again until an administrator has inspected it.
