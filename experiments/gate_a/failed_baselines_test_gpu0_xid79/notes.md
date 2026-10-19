# Failed run: Gate A dense baseline on GPU 0 (2026-10-14)

- Status: `failed`. No metrics were produced. Kept as failure evidence.
- What happened: about 66 s into the run, while SentenceTransformer was embedding the 482,105-product catalog
  (all-MiniLM-L6-v2 at `1110a243`, batch 256, fp32, GPU 0), CUDA raised `unspecified launch failure`.
  The kernel logged `NVRM: Xid 79, GPU has fallen off the bus` for PCI 0000:01:00 (GPU 0), then
  `Xid 154 ... Node Reboot Required` for both GPUs (01:00 and 21:00). See `dmesg_nvrm.txt`.
- Context: the same GPU logged Xid 31 (MMU fault) and Xid 43 under other users' processes about two days earlier.
  Another user's job was running on GPU 1 at launch; it was no longer running afterwards.
- Cause: unknown. Xid 79 usually means a PCIe, power, or thermal fault, not a software bug, but a
  hardware fault triggered under this load can't be ruled out.
- Consequence: all GPU work on gpu-host is paused until an administrator reboots the machine
  and the GPU passes a health check. The quality baseline was rerun on CPU instead (see `baselines_test_cpu`).
