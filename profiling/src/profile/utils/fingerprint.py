"""Capture static system metadata for profiling runs."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict


def capture_system_fingerprint() -> Dict[str, Any]:
    """Collect lightweight CPU/GPU/software metadata for the current node."""

    info: Dict[str, Any] = {
        "cpu_arch": platform.machine(),
        "cpu_cores": _safe_exec(["nproc"]),
        "kernel": platform.release(),
        "python_version": platform.python_version(),
        "cuda_version": _extract_cuda_version(),
    }

    gpu_query = "index,name,driver_version,memory.total,pstate,compute_mode"
    info["gpu_info"] = _safe_exec(
        [
            "nvidia-smi",
            f"--query-gpu={gpu_query}",
            "--format=csv,noheader",
        ]
    )

    return info


def persist_fingerprint(destination: Path) -> Path:
    """Write the current fingerprint JSON next to profiler outputs."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    data = capture_system_fingerprint()
    destination.write_text(json.dumps(data, indent=2, sort_keys=True))
    return destination


def _safe_exec(cmd: Any) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # pragma: no cover - defensive fallback
        return "unknown"


def _extract_cuda_version() -> str:
    output = _safe_exec(["nvcc", "--version"])
    if "release" in output:
        return output.split("release")[-1].strip().split(",")[0]
    return output or "unknown"


__all__ = ["capture_system_fingerprint", "persist_fingerprint"]
