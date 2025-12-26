#!/usr/bin/env python3
"""Generate the publication-ready normalized runtime plot for fan-out sweeps."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import csv
import re
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

BASE_INPUT = "denovo_1n0r_3prs"
RESULTS_ROOT = Path("results/denovo") / BASE_INPUT
MANIFEST_PATH = RESULTS_ROOT / "multi_run_manifest_v2.jsonl"
PLOTS_DIR = RESULTS_ROOT / "plots"
# Do not overwrite the existing figure; emit a dedicated filename.
OUTPUT_PATH = PLOTS_DIR / "fanout.png"
LEGEND_PATH = PLOTS_DIR / "fanout_legend.png"
PERCENT_CSV_PATH = PLOTS_DIR / "fanout.csv"
COLOR_PALETTE_PATH = Path(__file__).resolve().parent / "colors_fanout.yaml"

STAGE_LABELS = {
    "STRUCTURE_DESIGN": "RFDiffusion",
    "INVERSE_FOLDING": "ProteinMPNN",
    "VARIANT_GEN": "ESM-2",
    "MSA": "MMseqs2",
    "STRUCTURE_PREDICTION": "Protenix",
    "DOCK_LIGAND": "Vina-GPU",
}
STAGE_ORDER = [
    "RFDiffusion",
    "ProteinMPNN",
    "ESM-2",
    "MMseqs2",
    "Protenix",
    "Vina-GPU",
]
CANONICAL_COMPONENTS = {
    "RFDiffusion": "rfdiffusion",
    "ProteinMPNN": "proteinmpnn",
    "ESM-2": "esm-2",
    "MMseqs2": "mmseqs2",
    "Protenix": "protenix",
    "Vina-GPU": "vina_gpu",
}

PREFERRED_ORDER = [
    "rfdiffusion",
    "proteinmpnn",
    "esm",
    "mmseqs",
    "protenix",
    "esmfold",
    "diffdock",
    "vina_gpu",
    "vina_cpu",
]


def canonicalise(name: str) -> str:
    return re.sub(r"[^a-z]", "", str(name).lower())


def resolve_stage_colors() -> Dict[str, str]:
    """Load the component palette dedicated to fan-out plots."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to load component colors; install it before running plot_fanout.py.")

    palette_path = COLOR_PALETTE_PATH
    if not palette_path.exists():
        raise FileNotFoundError(
            f"Component color palette not found at {palette_path}; regenerate it or provide the file."
        )

    try:
        raw_data: Any = yaml.safe_load(palette_path.read_text())
    except Exception as exc:  # pragma: no cover - defensive, surfaces configuration errors
        raise RuntimeError(f"Failed to parse component colors from {palette_path}: {exc}") from exc

    if not raw_data:
        raise ValueError(f"Component color palette file {palette_path} is empty.")

    palette = raw_data.get("components") if isinstance(raw_data, dict) and "components" in raw_data else raw_data
    if not isinstance(palette, dict):
        raise ValueError(
            f"Component color palette in {palette_path} must be a mapping or include a 'components' mapping."
        )

    external: Dict[str, str] = {}
    for key, value in palette.items():
        external[canonicalise(key)] = str(value)

    components = [canonicalise(CANONICAL_COMPONENTS[stage]) for stage in STAGE_ORDER]

    desired = [comp for comp in PREFERRED_ORDER if comp in components]
    remaining = [comp for comp in components if comp not in desired]
    comp_order = desired + remaining

    comp_color_map: Dict[str, str] = {}
    missing: List[str] = []
    for comp in comp_order:
        if comp in external:
            comp_color_map[comp] = external[comp]
        else:
            missing.append(comp)

    if missing:
        raise KeyError(
            f"Missing component colors in {palette_path} for: " + ", ".join(sorted(set(missing)))
        )

    stage_colors: Dict[str, str] = {}
    for stage in STAGE_ORDER:
        canon = canonicalise(CANONICAL_COMPONENTS[stage])
        stage_colors[stage] = comp_color_map[canon]
    return stage_colors


STAGE_COLORS = resolve_stage_colors()
INCLUDED_STAGES = set(STAGE_ORDER)
FANOUT_LEVELS = (1, 2, 3)
COLLOCATE = False
GPU_COUNT = 1
TIME_COLUMN = "realtime"
BAR_HEIGHT = 0.25
BAR_SPACING = 0.5


def parse_duration(duration_str: str) -> float:
    total = 0.0
    for token in duration_str.split():
        if token.endswith("ms"):
            total += float(token[:-2]) / 1000.0
        elif token.endswith("s"):
            total += float(token[:-1])
        elif token.endswith("m"):
            total += float(token[:-1]) * 60.0
        elif token.endswith("h"):
            total += float(token[:-1]) * 3600.0
    return total


def normalise_stage(task_name: str) -> str:
    parts = task_name.split(":")
    stage_key = parts[2] if len(parts) > 2 else parts[-1]
    stage_key = stage_key.strip()
    label = STAGE_LABELS.get(stage_key)
    if label:
        return label
    tail = parts[-1]
    if " " in tail:
        tail = tail.split(" ", 1)[0]
    return tail


def load_manifest() -> Iterable[Dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    with MANIFEST_PATH.open() as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            yield json.loads(raw)


def resolve_trace(entry: Mapping) -> Optional[Path]:
    if entry.get("trace_file"):
        candidate = Path(entry["trace_file"])
        if candidate.exists():
            return candidate
    outdir = Path(entry["outdir"])
    trace_dir = outdir / "pipeline_info"
    traces = sorted(trace_dir.glob("execution_trace_*.txt"))
    return traces[-1] if traces else None


def parse_trace(trace_path: Path) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    with trace_path.open() as handle:
        reader = csv_reader(handle)
        for row in reader:
            stage = row["stage"]
            duration = row["duration_sec"]
            totals[stage] += duration
    return totals


def csv_reader(handle):
    import csv

    reader = csv.DictReader(handle, delimiter="\t")
    for row in reader:
        stage = normalise_stage(row["name"])
        if stage not in INCLUDED_STAGES:
            continue
        yield {
            "stage": stage,
            "duration_sec": parse_duration(row[TIME_COLUMN]),
        }


def collect_durations(entries: Iterable[Dict]) -> Dict[Tuple[int, int], Dict[str, float]]:
    aggregated: Dict[Tuple[int, int], Dict[str, float]] = {}
    for entry in entries:
        if (
            entry.get("status") != "success"
            or entry.get("fanout_level") not in FANOUT_LEVELS
            or bool(entry.get("collocate")) != COLLOCATE
            or int(entry.get("gpu_count")) != GPU_COUNT
        ):
            continue
        trace_path = resolve_trace(entry)
        if not trace_path:
            continue
        key = (int(entry["fanout_level"]), int(entry["run_index"]))
        aggregated[key] = parse_trace(trace_path)
    return aggregated


def summarise_trimmed_means(
    aggregated: Mapping[Tuple[int, int], Mapping[str, float]],
    stage_order: Sequence[str],
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, List[int]]]:
    per_fanout: Dict[int, List[Tuple[int, float, Dict[str, float]]]] = {}
    for (fanout, run), durations in aggregated.items():
        total = sum(durations.get(stage, 0.0) for stage in stage_order)
        per_fanout.setdefault(fanout, []).append((run, total, durations))

    summarised: Dict[int, Dict[str, float]] = {}
    selected_runs: Dict[int, List[int]] = {}

    for fanout, runs in per_fanout.items():
        if not runs:
            continue
        runs_sorted = sorted(runs, key=lambda item: item[1], reverse=True)
        subset = runs_sorted[:5]
        trimmed = subset[1:-1] if len(subset) >= 5 else subset
        if not trimmed:
            trimmed = runs_sorted
        count = len(trimmed)
        if count == 0:
            continue
        selected_runs[fanout] = [run for (run, _, _) in trimmed]
        stage_means: Dict[str, float] = {}
        for stage in stage_order:
            stage_means[stage] = sum(data.get(stage, 0.0) for (_, _, data) in trimmed) / count
        summarised[fanout] = stage_means

    return summarised, selected_runs


def to_minutes(values: Mapping[int, Dict[str, float]], stage_order: Sequence[str]) -> Dict[int, List[float]]:
    minutes: Dict[int, List[float]] = {}
    for fanout, stages in values.items():
        minutes[fanout] = [stages.get(stage, 0.0) / 60.0 for stage in stage_order]
    return minutes


def to_percentages(minutes: Mapping[int, List[float]]) -> Dict[int, List[float]]:
    percentages: Dict[int, List[float]] = {}
    for fanout, values in minutes.items():
        total = sum(values)
        if total > 0:
            percentages[fanout] = [value / total * 100.0 for value in values]
        else:
            percentages[fanout] = [0.0 for _ in values]
    return percentages


def write_percentages_csv(
    percentages: Mapping[int, List[float]], stage_order: Sequence[str]
) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    with PERCENT_CSV_PATH.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fanout_level", *stage_order])
        for fanout in sorted(percentages):
            row = [fanout] + [f"{value:.2f}" for value in percentages[fanout]]
            writer.writerow(row)


def style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")
    ax.tick_params(axis="both", labelsize=10)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)


def plot_horizontal(values: Mapping[int, List[float]], stage_order: Sequence[str]) -> List[Tuple[plt.Artist, str]]:
    fanouts = sorted(values)
    totals = [sum(values[fanout]) for fanout in fanouts]
    y_pos = [idx * BAR_SPACING for idx in range(len(fanouts))]

    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)
    fig, ax = plt.subplots(figsize=(6.0, 1.6))
    left_offsets = [0.0] * len(fanouts)
    stage_positions: Dict[str, List[Dict[str, float]]] = {stage: [] for stage in stage_order}
    legend_handles: List[Tuple[plt.Artist, str]] = []

    for idx, stage in enumerate(stage_order):
        widths: List[float] = []
        for j, fanout in enumerate(fanouts):
            total = totals[j]
            value = values[fanout][idx]
            width = (value / total * 100.0) if total else 0.0
            widths.append(width)
        if not any(widths):
            # still record left positions so that connectors align even if contribution drops to zero
            for j in range(len(fanouts)):
                stage_positions[stage].append({
                    "left": left_offsets[j],
                    "width": widths[j],
                    "y": y_pos[j],
                })
            continue
        bars = ax.barh(
            y_pos,
            widths,
            left=left_offsets,
            height=BAR_HEIGHT,
            color=STAGE_COLORS.get(stage, "#999999"),
            edgecolor="black",
            linewidth=0.4,
            label=stage,
        )
        if bars:
            legend_handles.append((bars[0], stage))
        for j, width in enumerate(widths):
            stage_positions[stage].append(
                {
                    "left": left_offsets[j],
                    "width": width,
                    "y": y_pos[j],
                }
            )
        left_offsets = [left + width for left, width in zip(left_offsets, widths)]

    bar_height = BAR_HEIGHT
    for stage, positions in stage_positions.items():
        if len(positions) < 2:
            continue
        for idx in range(len(positions) - 1):
            cur = positions[idx]
            nxt = positions[idx + 1]
            if cur["width"] <= 0 and nxt["width"] <= 0:
                continue
            x1 = cur["left"]
            y1 = cur["y"] + bar_height / 2.0
            x2 = nxt["left"]
            y2 = nxt["y"] - bar_height / 2.0
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="black",
                linewidth=0.7,
                alpha=0.8,
                linestyle=(0, (2, 4)),
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(["" for _ in fanouts])
    if y_pos:
        vertical_margin = BAR_HEIGHT * 0.5
        lower = y_pos[0] - BAR_HEIGHT / 2.0 - vertical_margin
        upper = y_pos[-1] + BAR_HEIGHT / 2.0 + vertical_margin
        ax.set_ylim(lower, upper)
    ax.invert_yaxis()
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.88))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)

    return legend_handles


def main() -> None:
    entries = list(load_manifest())
    aggregated = collect_durations(entries)
    if not aggregated:
        raise SystemExit("No successful runs matching the paper plot criteria.")

    trimmed, selected_runs = summarise_trimmed_means(aggregated, STAGE_ORDER)
    if not trimmed:
        raise SystemExit("Insufficient successful runs to compute trimmed means.")

    minutes = to_minutes(trimmed, STAGE_ORDER)
    legend_handles = plot_horizontal(minutes, STAGE_ORDER)
    percentages = to_percentages(minutes)
    write_percentages_csv(percentages, STAGE_ORDER)

    if legend_handles:
        legend_fig, legend_ax = plt.subplots(figsize=(5.5, 0.6))
        legend_ax.axis("off")
        handles, labels = zip(*legend_handles)
        legend_fig.legend(
            handles,
            labels,
            loc="center",
            ncol=len(handles),
            frameon=False,
            columnspacing=0.8,
            handlelength=1.6,
            handletextpad=0.4,
        )
        legend_fig.tight_layout()
        legend_fig.savefig(LEGEND_PATH, dpi=300, transparent=True)
        plt.close(legend_fig)

    for fanout in sorted(selected_runs):
        runs = ", ".join(str(idx) for idx in selected_runs[fanout])
        print(f"Fanout {fanout} trimmed runs: [{runs}]")
    print(f"Saved plot to {OUTPUT_PATH}")
    if legend_handles:
        print(f"Saved legend to {LEGEND_PATH}")
    print(f"Saved normalized percentages to {PERCENT_CSV_PATH}")


if __name__ == "__main__":
    main()
