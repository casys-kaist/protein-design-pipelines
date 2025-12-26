#!/usr/bin/env python3
"""Run AutoDock Vina-GPU repeatedly to honor the input batch knob."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch runner for AutoDock Vina-GPU.")
    parser.add_argument(
        "--binary",
        default="/vina/AutoDock-Vina-GPU-2.1/AutoDock-Vina-GPU-2-1",
        help="Path to the Vina-GPU binary.",
    )
    parser.add_argument(
        "--opencl_binary_path",
        default="/vina/AutoDock-Vina-GPU-2.1",
        help="Directory containing OpenCL binaries (passed through to Vina-GPU).",
    )
    parser.add_argument("--receptor", required=True, help="Receptor PDBQT path.")
    parser.add_argument("--ligand", required=True, help="Ligand PDBQT path.")
    parser.add_argument("--out", required=True, help="Base output PDBQT path.")
    parser.add_argument("--center_x", required=True, help="Docking box center (x).")
    parser.add_argument("--center_y", required=True, help="Docking box center (y).")
    parser.add_argument("--center_z", required=True, help="Docking box center (z).")
    parser.add_argument("--size_x", default="25", help="Docking box size (x).")
    parser.add_argument("--size_y", default="25", help="Docking box size (y).")
    parser.add_argument("--size_z", default="25", help="Docking box size (z).")
    parser.add_argument(
        "--thread",
        default="8000",
        help="Thread count for Vina-GPU (kept as string to avoid formatting surprises).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of times to invoke Vina-GPU for the given ligand.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace, out_path: Path) -> List[str]:
    return [
        str(args.binary),
        "--opencl_binary_path",
        str(args.opencl_binary_path),
        "--receptor",
        str(args.receptor),
        "--ligand",
        str(args.ligand),
        "--out",
        str(out_path),
        "--center_x",
        str(args.center_x),
        "--center_y",
        str(args.center_y),
        "--center_z",
        str(args.center_z),
        "--size_x",
        str(args.size_x),
        "--size_y",
        str(args.size_y),
        "--size_z",
        str(args.size_z),
        "--thread",
        str(args.thread),
    ]


def _format_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def main() -> int:
    args = parse_args()
    batch_size = max(1, int(args.batch_size))
    base_out = Path(args.out)
    base_out.parent.mkdir(parents=True, exist_ok=True)

    failures = []
    for idx in range(batch_size):
        suffix = "" if idx == 0 else f"_b{idx + 1}"
        out_path = base_out if not suffix else base_out.with_name(f"{base_out.stem}{suffix}{base_out.suffix}")
        cmd = build_command(args, out_path)
        print(f"[vina_gpu_batch] {idx + 1}/{batch_size}: {_format_cmd(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime feedback only
            failures.append((idx + 1, exc.returncode))

    if failures:
        for run_idx, code in failures:
            print(f"[vina_gpu_batch] run {run_idx} failed with exit code {code}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
