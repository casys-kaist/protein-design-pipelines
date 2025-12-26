"""Concrete profiler strategy implementations used by Hydra configs."""

from __future__ import annotations

import csv
import datetime as dt
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:  # Optional dependency; raise at runtime when telemetry is requested.
    import psutil
except ModuleNotFoundError:  # pragma: no cover - fallback for test environments
    psutil = None

from .base import ProfilerStrategy


class TelemetryMonitor:
    """Background sampler that records lightweight telemetry to CSV."""

    def __init__(
        self,
        *,
        interval: float,
        output_file: Path,
        gpu_query: str,
    ) -> None:
        self.interval = max(interval, 0.1)
        self.output_file = Path(output_file)
        self.gpu_query = gpu_query
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="telemetry-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run(self) -> None:
        header = self._build_header()
        with self.output_file.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            handle.flush()
            while not self._stop.is_set():
                row = self._collect_row()
                writer.writerow(row)
                handle.flush()
                if self._stop.wait(self.interval):
                    break

    def _build_header(self) -> List[str]:
        header = [
            "timestamp",
            "cpu_percent",
            "mem_percent",
        ]
        gpu_count = len(self._collect_gpu()) or 1
        for idx in range(gpu_count):
            header.extend(
                [
                    f"gpu{idx}_name",
                    f"gpu{idx}_util",
                    f"gpu{idx}_mem_util",
                    f"gpu{idx}_mem_total",
                    f"gpu{idx}_mem_used",
                ]
            )
        return header

    def _collect_row(self) -> List[Any]:
        if psutil is None:
            raise RuntimeError("psutil is required for Level A telemetry; install via `pip install psutil`.")
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        gpu_rows = self._collect_gpu()
        if gpu_rows:
            # Drop the leading GPU index when the default query is used so the
            # CSV columns line up with the header (name, util, mem_util, total, used).
            cleaned: List[List[str]] = []
            for row in gpu_rows:
                if len(row) >= 6 and str(row[0]).strip().isdigit():
                    cleaned.append(row[1:])
                else:
                    cleaned.append(row)
            gpu_rows = cleaned
        else:
            gpu_rows = [["n/a", "0", "0", "0", "0"]]
        flat: List[Any] = [dt.datetime.utcnow().isoformat(), f"{cpu:.2f}", f"{mem:.2f}"]
        for row in gpu_rows:
            flat.extend(row)
        return flat

    def _collect_gpu(self) -> List[List[str]]:
        if shutil.which("nvidia-smi") is None:
            return []
        query = self.gpu_query or "index,name,utilization.gpu,utilization.memory,memory.total,memory.used"
        cmd = [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []
        rows: List[List[str]] = []
        for line in output.strip().splitlines():
            if not line.strip():
                continue
            chunks = [chunk.strip() for chunk in line.split(",")]
            rows.append(chunks)
        return rows


class LevelATelemetryStrategy(ProfilerStrategy):
    """Level A strategy using lightweight telemetry sampling."""

    def __init__(self, **config: Any) -> None:
        super().__init__(**config)
        self._monitor: Optional[TelemetryMonitor] = None

    def build_command(self, original_cmd: List[str], output_path: str) -> List[str]:
        return list(original_cmd)

    def pre_run_hook(self, meta: Dict[str, Any]) -> None:
        interval = float(self.config.get("sampling_interval", 0.5))
        gpu_query = self.config.get(
            "gpu_query",
            "index,name,utilization.gpu,utilization.memory,memory.total,memory.used",
        )
        # Always emit telemetry into the raw profiler directory (meta["raw_dir"])
        # so artifacts live in a single canonical location. Fall back to other
        # hints only if raw_dir is unavailable.
        output_name = self.config.get("output_filename") or f"{meta.get('run_id', 'run')}_telemetry.csv"
        base_dir = meta.get("raw_dir") or meta.get("prof_data_dir") or meta.get("prof_dir") or meta.get("run_dir")
        base_path = Path(base_dir) if base_dir else Path(".")
        telemetry_path = Path(meta.get("output_path") or (base_path / output_name))
        self._monitor = TelemetryMonitor(
            interval=interval,
            output_file=telemetry_path,
            gpu_query=gpu_query,
        )
        self._monitor.start()
        meta["telemetry_file"] = str(telemetry_path)

    def post_run_hook(self, meta: Dict[str, Any]) -> None:
        if self._monitor:
            self._monitor.stop()
            self._monitor = None


class NsysProfilerStrategy(ProfilerStrategy):
    """Level B profiler using Nsight Systems timeline traces."""

    def build_command(self, original_cmd: List[str], output_path: str) -> List[str]:
        base = _strip_extension(Path(output_path), ".nsys-rep")
        # Use separate arg form to avoid Nsight interpreting "-o=<path>" as a relative "=<path>".
        cmd = ["nsys", "profile", "-o", str(base)]
        for param in self.config.get("params", []):
            cmd.append(str(param))
        cmd.extend(original_cmd)
        return cmd

    def pre_run_hook(self, meta: Dict[str, Any]) -> None:  # pragma: no cover - intentionally empty
        return None

    def post_run_hook(self, meta: Dict[str, Any]) -> None:  # pragma: no cover - intentionally empty
        return None


class NcuProfilerStrategy(ProfilerStrategy):
    """Level C profiler using Nsight Compute for kernel drill-down."""

    def build_command(self, original_cmd: List[str], output_path: str) -> List[str]:
        path = _strip_extension(Path(output_path), ".ncu-rep")
        cmd = ["ncu", "-o", str(path), "-f"]
        params = self.config.get("params") or ["--target-processes", "all"]
        for param in params:
            cmd.append(str(param))
        kernel_regex = self.config.get("kernel_regex")
        if kernel_regex:
            cmd.extend(["--kernel-name", kernel_regex])
        metrics = self.config.get("metrics") or []
        for metric in metrics:
            cmd.extend(["--metrics", metric])
        sections = self.config.get("target_sections") or []
        for section in sections:
            cmd.extend(["--section", section])
        cmd.extend(original_cmd)
        return cmd

    def pre_run_hook(self, meta: Dict[str, Any]) -> None:  # pragma: no cover - intentionally empty
        return None

    def post_run_hook(self, meta: Dict[str, Any]) -> None:  # pragma: no cover - intentionally empty
        return None


def _strip_extension(path: Path, suffix: str) -> Path:
    if suffix and path.name.endswith(suffix):
        return path.with_name(path.name[: -len(suffix)])
    return path


__all__ = [
    "LevelATelemetryStrategy",
    "NcuProfilerStrategy",
    "NsysProfilerStrategy",
    "TelemetryMonitor",
]
