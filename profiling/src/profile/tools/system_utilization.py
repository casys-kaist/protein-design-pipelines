#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
System + multi-GPU monitor using NVML (pynvml) + psutil.

Writes a single CSV: <PROF_DIR>/<BASENAME>_utilization.csv
Each row = one sampling timepoint:
  elapsed_s, cpu_percent_system, mem_percent_system,
  gpu0_sm_percent, gpu0_mem_percent, gpu0_vram_used_MiB, gpu0_vram_total_MiB,
  gpu1_sm_percent, gpu1_mem_percent, gpu1_vram_used_MiB, gpu1_vram_total_MiB,
  ...
"""

from __future__ import annotations
import os
import csv
import time
import psutil
import subprocess
from typing import Optional

# NVML (pynvml)
try:
    import pynvml as nvml
except Exception as e:
    raise RuntimeError(
        "Failed to import pynvml. Install with: pip install nvidia-ml-py3"
    ) from e

# ===== USER SECTION =====

CMD = ['python3', '-m', 'inference']
PARAMS = [
    '--protein_path', '/workspace/profiling/sample_input/esmfold_result.pdb',
    '--ligand_description', '/workspace/profiling/sample_input/1yc1_ligand.mol2',
    '--out_dir', '/workspace/profiling/diffdock/output',
    '--complex_name', 'result'
]
PROF_DIR = "/workspace/profiling/diffdock"
BASENAME = "diffdock"

# Optional preparation commands
PREP_COMMANDS = [
    'rm -rf /workspace/profiling/diffdock/output'
]

# Max retry attempts when the profiled command exits non-zero; 0 means unlimited (legacy).
MAX_ATTEMPTS = int(os.environ.get("SYSTEM_UTIL_MAX_ATTEMPTS", "0"))

# Sampling interval (seconds)
SAMPLE_INTERVAL = 0.1

# ========================

# ============================
# Utilities
# ============================
def which(cmd: str) -> Optional[str]:
    from shutil import which as _which
    return _which(cmd)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def csv_path() -> str:
    return os.path.join(PROF_DIR, f"{BASENAME}_utilization.csv")

def build_header(ngpu: int):
    """Build CSV header: system columns + per-GPU columns."""
    header = ["elapsed_s", "cpu_percent_system", "mem_percent_system"]
    for i in range(ngpu):
        header += [
            f"gpu{i}_sm_percent",
            f"gpu{i}_mem_percent",
            f"gpu{i}_vram_used_MiB",
            f"gpu{i}_vram_total_MiB"
        ]
    return header

def run_prep_commands(cmds):
    for c in cmds:
        try:
            subprocess.run(c, shell=True, check=False)
        except Exception:
            pass

# ============================
# Monitor
# ============================
def run_and_monitor_nvml():
    # init NVML once
    nvml.nvmlInit()
    try:
        try:
            ngpu = nvml.nvmlDeviceGetCount()
            if ngpu == 0:
                raise RuntimeError("No NVIDIA GPU detected by NVML.")
        except Exception:
            nvml.nvmlShutdown()
            raise

        ensure_dir(PROF_DIR)
        out_csv = csv_path()

        attempt = 1
        while True:
            print(f"[INFO] ===== Attempt {attempt} =====")

            # reset CSV and write header
            with open(out_csv, "w", newline="") as f:
                csv.writer(f).writerow(build_header(ngpu))

            # run prep every attempt; on failure retry
            try:
                if PREP_COMMANDS:
                    run_prep_commands(PREP_COMMANDS)
            except Exception as e:
                print(f"[ERROR] PREP failed: {e}. Retrying...")
                attempt += 1
                continue

            full_cmd = CMD + PARAMS
            print(f"[INFO] Starting command: {' '.join(full_cmd)}")
            print(f"[INFO] Output CSV: {out_csv}")
            print(f"[INFO] Sampling interval: {SAMPLE_INTERVAL}s")

            _ = psutil.cpu_percent(interval=None)  # prime

            try:
                try:
                    p = subprocess.Popen(full_cmd)
                except FileNotFoundError as e:
                    print(f"[ERROR] Command not found: {e}. Retrying...")
                    attempt += 1
                    continue

                t0 = time.perf_counter()
                with open(out_csv, "a", newline="") as f:
                    writer = csv.writer(f)

                    while True:
                        if p.poll() is not None:
                            ret = p.returncode
                            if ret != 0:
                                print(f"[ERROR] Process exited with code {ret}. Retrying...")
                                attempt += 1
                                break  # go to next attempt (CSV already reset next loop)
                            elapsed = time.perf_counter() - t0
                            writer.writerow(sample_row(ngpu, elapsed))
                            print(f"[INFO] Duration (s): {elapsed:.3f}")
                            print(f"[INFO] Wrote: {out_csv}")
                            return  # success

                        elapsed = time.perf_counter() - t0
                        writer.writerow(sample_row(ngpu, elapsed))
                        time.sleep(SAMPLE_INTERVAL)

            except Exception as e:
                print(f"[ERROR] Unexpected error: {e}. Retrying...")
                attempt += 1
                continue

            attempt += 1
            if MAX_ATTEMPTS and attempt > MAX_ATTEMPTS:
                raise RuntimeError(f"Aborting after {attempt - 1} failed attempt(s)")

    finally:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass

def sample_row(ngpu: int, elapsed: float):
    """
    Collect system + per-GPU metrics into a single flat row.
    """
    row = [f"{elapsed:.3f}"]

    # System metrics
    cpu_pct = psutil.cpu_percent(interval=None)       # %
    mem_pct = psutil.virtual_memory().percent         # %
    row += [f"{cpu_pct:.2f}", f"{mem_pct:.2f}"]

    # Per-GPU metrics
    for i in range(ngpu):
        try:
            h = nvml.nvmlDeviceGetHandleByIndex(i)
            util = nvml.nvmlDeviceGetUtilizationRates(h)
            sm = float(util.gpu)
            mem = float(util.memory)

            meminfo = nvml.nvmlDeviceGetMemoryInfo(h)
            used_mib = int(meminfo.used / (1024 * 1024))
            total_mib = int(meminfo.total / (1024 * 1024))

            row += [f"{sm:.2f}", f"{mem:.2f}", used_mib, total_mib]
        except Exception:
            row += ["", "", "", ""]
    return row

if __name__ == "__main__":
    run_and_monitor_nvml()
