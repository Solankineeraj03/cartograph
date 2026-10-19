"""Run manifests: code revision, environment, hardware, inputs, and timing."""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _run(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state() -> dict:
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    diff = _run(["git", "diff", "HEAD"])
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status) if status is not None else None,
        "dirty_files": status.splitlines() if status else [],
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
    }


def environment() -> dict:
    freeze = _run([sys.executable, "-m", "pip", "freeze"]) or _run(
        ["uv", "pip", "freeze", "--python", sys.executable])
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "pip_freeze": freeze.splitlines() if freeze else None,
        "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest() if freeze else None,
    }


def hardware() -> dict:
    info = {"hostname": platform.node(), "cpu_count": os.cpu_count()}
    gpus = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,pci.bus_id",
                 "--format=csv,noheader"])
    info["nvidia_smi"] = gpus.splitlines() if gpus else None
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
            info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return info


class Run:
    """Context manager that writes manifest.json (also on failure) into an experiment directory."""

    def __init__(self, out_dir: Path, name: str, config: dict, inputs: dict[str, Path] | None = None,
                 seed: int | None = None, status: str = "measured_local", lock: Path | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        lock_record = None
        if lock is not None:
            lock_path = ROOT / lock
            lock_record = {"path": str(lock), "sha256": None, "present": lock_path.exists()}
            if lock_path.exists():
                (self.out_dir / "environment-lock.txt").write_bytes(lock_path.read_bytes())
                lock_record["sha256"] = sha256_file(lock_path)
        self.manifest = {
            "experiment": name, "status": status, "seed": seed, "config": config,
            "command": [sys.executable, *sys.argv], "cwd": os.getcwd(),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "git": git_state(), "environment": environment(), "hardware": hardware(),
            "inputs": {key: {"path": str(path), "sha256": sha256_file(Path(path))}
                       for key, path in (inputs or {}).items()},
            "environment_lock": lock_record,
            "extra": {},
        }
        (self.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, kind, value, traceback):
        self.manifest["duration_seconds"] = round(time.perf_counter() - self._start, 3)
        self.manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if kind is not None:
            self.manifest["status"] = "failed"
            self.manifest["error"] = f"{kind.__name__}: {value}"
        outputs = {}
        for path in sorted(self.out_dir.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                outputs[str(path.relative_to(self.out_dir))] = sha256_file(path)
        self.manifest["outputs_sha256"] = outputs
        (self.out_dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n")
        return False
