#!/usr/bin/env python3
"""Plot component runtimes for OBS#2 fan-out experiments."""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.modules.pop("profile", None)
from profile import component_output_root

DEFAULT_OUTPUT = component_output_root("obs2") / "fanout_breakdown_try1-3.png"

STAGE_ORDER = [
    "RFDiffusion",
    "ProteinMPNN",
    "ESM",
    "MMseqs2",
    "Protenix",
    "Docking",
    "MultiQC",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/denovo/denovo_1n0r_3prs"),
        help="Root directory that holds fanout_* folders.",
    )
    parser.add_argument(
        "--collocation",
        default="exclusive",
        help="Collocation directory name (e.g., exclusive, shared).",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=1,
        help="GPU count segment to inspect (e.g., 1 for gpu_1).",
    )
    parser.add_argument(
        "--fanout-levels",
        type=int,
        nargs="*",
        default=[1, 2, 3],
        help="Fan-out levels to load.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="*",
        default=[1, 2, 3],
        help="Run indices to plot (interpreted as try1/try2/try3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to save the figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure once rendering completes.",
    )
    return parser.parse_args()


_DURATION_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*([a-z]+)")
_UNIT_SCALE = {
    "d": 86400.0,
    "h": 3600.0,
    "m": 60.0,
    "s": 1.0,
    "ms": 0.001,
}


def parse_duration(value: str) -> float:
    if not value:
        return 0.0
    value = value.strip()
    if not value:
        return 0.0
    total = 0.0
    for number, unit in _DURATION_PATTERN.findall(value):
        scale = _UNIT_SCALE.get(unit.lower())
        if scale is None:
            continue
        total += float(number) * scale
    return total


def stage_from_task(name: str) -> str:
    if "RUN_RFDIFFUSION" in name:
        return "RFDiffusion"
    if "RUN_PROTEINMPNN" in name:
        return "ProteinMPNN"
    if "RUN_ESM" in name:
        return "ESM"
    if "RUN_MMSEQS2" in name:
        return "MMseqs2"
    if "RUN_PROTENIX" in name or "CONVERT_CIF_TO_PDB" in name:
        return "Protenix"
    if (
        "RUN_VINA_GPU" in name
        or "RUN_VINA" in name
        or "RUN_AUTODOCK" in name
        or "CONVERT_PDB_TO_PDBQT" in name
    ):
        return "Docking"
    if "MULTIQC" in name:
        return "MultiQC"
    return ""


def load_trace(trace_path: Path) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    with trace_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            stage = stage_from_task(row.get("name", ""))
            if not stage:
                continue
            duration = parse_duration(row.get("realtime", ""))
            if duration == 0.0:
                duration = parse_duration(row.get("duration", ""))
            totals[stage] += duration
    return totals


def pick_trace_file(run_dir: Path) -> Path:
    trace_dir = run_dir / "pipeline_info"
    candidates = sorted(trace_dir.glob("execution_trace_*.txt"))
    if not candidates:
        raise FileNotFoundError(f"No trace file found under {trace_dir}")
    return candidates[-1]


def collect_stage_totals(
    base_dir: Path,
    fanout_levels: Sequence[int],
    runs: Sequence[int],
    collocation: str,
    gpu_count: int,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    results: Dict[Tuple[int, int], Dict[str, float]] = {}
    for level in fanout_levels:
        for run in runs:
            run_dir = (
                base_dir
                / f"fanout_{level}"
                / collocation
                / f"gpu_{gpu_count}"
                / f"run_{run}"
            )
            if not run_dir.exists():
                print(f"Warning: missing run directory {run_dir}")
                continue
            try:
                trace_path = pick_trace_file(run_dir)
            except FileNotFoundError as err:
                print(err)
                continue
            results[(level, run)] = load_trace(trace_path)
    return results


def plot_totals(
    totals: Dict[Tuple[int, int], Dict[str, float]],
    fanout_levels: Sequence[int],
    runs: Sequence[int],
    output_path: Path,
    show: bool,
) -> None:
    if not totals:
        raise RuntimeError("No stage totals were collected; nothing to plot.")

    fig, axes = plt.subplots(1, len(runs), sharey=True, figsize=(4 * len(runs), 4))
    if len(runs) == 1:
        axes = [axes]

    for ax, run in zip(axes, runs):
        bottoms = [0.0] * len(fanout_levels)
        x_positions = list(range(len(fanout_levels)))
        for stage in STAGE_ORDER:
            values = [
                totals.get((level, run), {}).get(stage, 0.0) / 60.0
                for level in fanout_levels
            ]
            if not any(values):
                continue
            ax.bar(x_positions, values, bottom=bottoms, label=stage)
            bottoms = [b + v for b, v in zip(bottoms, values)]

        ax.set_title(f"try{run}")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(level) for level in fanout_levels])
        ax.set_xlabel("fan-out level")
        if ax is axes[0]:
            ax.set_ylabel("wall-clock minutes")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles))
    fig.suptitle("OBS#2 pipeline component runtimes (no collocate, 1 GPU)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved figure to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    totals = collect_stage_totals(
        base_dir=args.base_dir,
        fanout_levels=args.fanout_levels,
        runs=args.runs,
        collocation=args.collocation,
        gpu_count=args.gpu_count,
    )
    plot_totals(
        totals=totals,
        fanout_levels=args.fanout_levels,
        runs=args.runs,
        output_path=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
