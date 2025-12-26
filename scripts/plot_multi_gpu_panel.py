#!/usr/bin/env python3
"""Render a 2x2 panel that combines GPU time-share medians and multi-GPU overlays."""

from __future__ import annotations

import argparse
import importlib.util
import os
from collections import OrderedDict
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

sns.set_theme(style="white", context="paper", font_scale=1.08)
plt.rcParams.update(
    {
        "axes.edgecolor": "0.2",
        "axes.linewidth": 0.8,
        "grid.color": "0.85",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "figure.dpi": 110,
        "xtick.major.pad": 2.0,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)

def _load_local_module(alias: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(alias, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module '{alias}' from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


SCRIPT_DIR = Path(__file__).resolve().parent
multi = _load_local_module("repo_plot_multi_gpu", SCRIPT_DIR / "plot_multi_gpu.py")
timeshare = _load_local_module("repo_plot_gpu_timeshare", SCRIPT_DIR / "plot_gpu_timeshare.py")


DEFAULT_BASE = multi.DEFAULT_BASE
DEFAULT_OUTPUT = DEFAULT_BASE / "plots" / "multi_gpu_panel"
DEFAULT_FIGURE_NAME = "multi_gpu_panel.png"
MULTI_LEGEND_NAME = "gpu_overlay_legend.png"
TIMESHARE_LEGEND_NAME = "gpu_timeshare_legend.png"


TimeshareLegendHandles = Dict[str, Tuple[int, Line2D]]


def summarize_multi_gpu_reduction(fanout_map: Dict[int, List[multi.RunMeanSeries]]) -> List[str]:
    """Return human-readable percent reduction summaries vs GPU1 baselines."""
    summaries: List[str] = []
    for fanout in sorted(fanout_map):
        runs = fanout_map[fanout]
        if not runs:
            continue
        lookup = {run.gpu_count: run for run in runs}
        baseline = lookup.get(1)
        if baseline is None or baseline.wall_clock_s <= 0:
            continue
        for gpu_count in sorted(lookup):
            if gpu_count == 1:
                continue
            run = lookup[gpu_count]
            reduction = (baseline.wall_clock_s - run.wall_clock_s) / baseline.wall_clock_s * 100.0
            summaries.append(f"fanout{fanout} / gpu{gpu_count}: {reduction:.1f}%")
    return summaries


def parse_median_override(spec: str) -> Tuple[int, str, int]:
    try:
        fanout_str, mode, run_str = spec.split(":", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid override '{spec}'. Expected format fanout:mode:run e.g. 2:exclusive:4"
        ) from exc
    try:
        fanout = int(fanout_str)
        run_idx = int(run_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Fan-out and run index must be integers in '{spec}'.") from exc
    mode_key = mode.strip().lower()
    if not mode_key:
        raise argparse.ArgumentTypeError(f"Mode must be non-empty in '{spec}'.")
    return fanout, mode_key, run_idx


def unique_preserve_order(values: Sequence[int]) -> List[int]:
    seen = set()
    ordered: List[int] = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def collect_timeshare_entries(
    *,
    base_dir: Path,
    fanouts: Sequence[int],
    modes: Sequence[str],
    gpu_count: int,
) -> Dict[int, List[timeshare.CombinedSeries]]:
    entries: Dict[int, List[timeshare.CombinedSeries]] = {}
    for fanout, mode, run_dir in timeshare.iter_runs(base_dir, fanouts, modes, gpu_count):
        csv_path = run_dir / "gpu_metrics.csv"
        if not csv_path.exists():
            continue
        try:
            elapsed, series = timeshare.load_gpu_metrics(csv_path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[multi_gpu_panel] Failed to read {csv_path}: {exc}")
            continue
        if not elapsed:
            continue
        max_curve = timeshare.build_max_curve(elapsed, series)
        wall_clock_s, avg_max_util = timeshare.compute_summary(elapsed, series)
        run_index = int(run_dir.name.split("_")[-1])
        final_outputs = timeshare.count_final_outputs(run_dir)
        entries.setdefault(fanout, []).append(
            timeshare.CombinedSeries(
                mode=mode.lower(),
                run_index=run_index,
                wall_clock_s=wall_clock_s,
                avg_max_util=avg_max_util,
                max_curve=max_curve,
                final_outputs=final_outputs,
                gpu_samples=len(elapsed),
            )
        )
        samples_per_hour = (
            (final_outputs / wall_clock_s) * 3600.0 if wall_clock_s > 0 and final_outputs > 0 else 0.0
        )
        print(
            "[multi_gpu_panel] Loaded time-share fan-out {fanout} {mode} run {run}: "
            "{duration:.2f} min, avg max util {util:.1f}%, {finals} outputs ({rate:.2f} samples/hour)".format(
                fanout=fanout,
                mode=mode,
                run=run_index,
                duration=wall_clock_s / 60.0,
                util=avg_max_util,
                finals=final_outputs,
                rate=samples_per_hour,
            )
        )
    return entries


def render_multi_subplot(
    ax: plt.Axes,
    *,
    fanout: int,
    runs: List[multi.RunMeanSeries],
    smooth_window: float,
    smooth_operation: str,
    smooth_passes: Optional[Sequence[Tuple[str, float]]],
    color_map: Dict[int, str],
    legend_handles: Dict[str, Line2D],
) -> bool:
    if not runs:
        return False

    plotted = False
    palette = sns.color_palette("colorblind", n_colors=max(len(runs), 3))
    for idx, run in enumerate(sorted(runs, key=lambda item: item.gpu_count)):
        smoothed = multi.apply_smoothing_pipeline(
            run.mean_curve,
            smooth_window,
            smooth_operation,
            smooth_passes,
        )
        if not smoothed:
            continue
        times = [t for t, _ in smoothed]
        utils = [util for _, util in smoothed]
        if not utils:
            continue
        color = color_map.get(run.gpu_count) or palette[idx % len(palette)]
        label = f"{run.gpu_count} GPU"
        (line,) = ax.plot(
            times,
            utils,
            color=color,
            linewidth=1.0,
            solid_capstyle="round",
            antialiased=True,
            label=label,
        )
        if times:
            ax.axvline(
                times[-1],
                color=color,
                linestyle="--",
                linewidth=1.3,
                alpha=0.9,
            )
        if label not in legend_handles:
            legend_handles[label] = line
        plotted = True

    if not plotted:
        return False

    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", linestyle="-", linewidth=0.6, color="0.85", alpha=0.7)
    multi.style_axes(ax)
    ax.set_ylabel("")
    return True


def render_timeshare_subplot(
    ax: plt.Axes,
    *,
    fanout: int,
    entries: List[timeshare.CombinedSeries],
    smooth_window: float,
    smooth_operation: str,
    smooth_passes: Optional[Sequence[Tuple[str, float]]],
    min_activity: float,
    median_overrides: Optional[Dict[Tuple[int, str], int]],
    legend_handles: TimeshareLegendHandles,
) -> bool:
    if not entries:
        return False

    median_entries = timeshare.select_median_entries(fanout, entries, median_overrides)
    if not median_entries:
        return False

    plotted = False
    fallback_palette = sns.color_palette("colorblind", n_colors=max(len(median_entries), 2))

    def _priority(entry: timeshare.CombinedSeries) -> int:
        return timeshare.MODE_RENDER_PRIORITY.get(entry.mode.lower(), 99)

    for idx, entry in enumerate(sorted(median_entries, key=_priority)):
        curve = timeshare.apply_smoothing_pipeline(
            entry.max_curve,
            smooth_window,
            smooth_operation,
            smooth_passes,
        )
        if not curve:
            continue
        times = [t for t, _ in curve]
        utils = [util for _, util in curve]
        if not utils or max(utils) < min_activity:
            continue
        mode_key = entry.mode.lower()
        style = timeshare.MODE_STYLE_MAP.get(mode_key, {})
        color = style.get("color", fallback_palette[idx % len(fallback_palette)])
        label = style.get("label", entry.mode.title())
        (line,) = ax.plot(
            times,
            utils,
            color=color,
            linewidth=1.0,
            solid_capstyle="round",
            antialiased=True,
            label=label,
        )
        if times:
            ax.axvline(
                times[-1],
                color=color,
                linestyle="--",
                linewidth=1.3,
                alpha=0.9,
            )
        plotted = True
        if label not in legend_handles:
            legend_handles[label] = (timeshare.MODE_RENDER_PRIORITY.get(mode_key, 99), line)

    if not plotted:
        return False

    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", linestyle="-", linewidth=0.6, color="0.85", alpha=0.7)
    multi.style_axes(ax)
    ax.set_ylabel("")
    return True


def save_timeshare_legend(handles: TimeshareLegendHandles, out_path: Path) -> Optional[Path]:
    if not handles:
        return None
    ordered = sorted(handles.items(), key=lambda item: (item[1][0], item[0]))
    legend_fig, legend_ax = plt.subplots(figsize=(3.0, 0.55))
    legend_ax.axis("off")
    legend_fig.legend(
        [line for _, (_, line) in ordered],
        [label for label, _ in ordered],
        loc="center",
        ncol=len(ordered),
        frameon=False,
        columnspacing=0.8,
        handlelength=1.6,
        handletextpad=0.6,
    )
    legend_fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    legend_fig.savefig(out_path, dpi=300, transparent=True)
    plt.close(legend_fig)
    return out_path


def resolve_manifest(base_dir: Path, manifest_path: Optional[Path]) -> Optional[Path]:
    if manifest_path is not None:
        expanded = manifest_path.expanduser()
        if expanded.exists():
            return expanded.resolve()
        print(f"[multi_gpu_panel] Manifest path '{manifest_path}' not found; status filtering disabled.")
        return None
    detected = multi.discover_manifest_path(base_dir)
    if detected:
        return detected.resolve()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE,
        help="Root directory that contains fanout_<k>/<mode>/gpu_<n>/run_* folders (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the combined panel and legend PNGs (default: <base>/plots/multi_gpu_panel)",
    )
    parser.add_argument(
        "--figure-name",
        type=str,
        default=DEFAULT_FIGURE_NAME,
        help="Filename for the combined 2x2 panel (default: %(default)s)",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional JSONL manifest used to confirm completed runs for the multi-GPU column (default: auto-detect)",
    )
    parser.add_argument(
        "--multi-fanouts",
        type=int,
        nargs="+",
        default=[4, 5],
        help="Fan-out levels to include in the multi-GPU column (default: %(default)s)",
    )
    parser.add_argument(
        "--multi-gpu-counts",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="GPU count labels to include for the multi-GPU column (default: %(default)s)",
    )
    parser.add_argument(
        "--multi-modes",
        type=str,
        nargs="+",
        default=["collocate"],
        help="Execution modes to inspect when selecting multi-GPU runs (default: %(default)s)",
    )
    parser.add_argument(
        "--multi-min-activity",
        type=float,
        default=1.0,
        help="Minimum GPU utilisation threshold when constructing mean curves (default: %(default)s)",
    )
    parser.add_argument(
        "--multi-color-config",
        type=Path,
        default=multi.DEFAULT_COLOR_CONFIG,
        help="YAML file mapping gpu<count> labels to hex colors (default: %(default)s)",
    )
    parser.add_argument(
        "--multi-smooth-window",
        type=float,
        default=multi.DEFAULT_SMOOTH_WINDOW,
        help="Base smoothing window (seconds) for the multi-GPU column (default: %(default)s)",
    )
    parser.add_argument(
        "--multi-smooth-operation",
        type=str,
        choices=["max", "mean"],
        default=multi.DEFAULT_SMOOTH_OPERATION,
        help="Base smoothing operation for the multi-GPU column (default: %(default)s)",
    )
    parser.add_argument(
        "--multi-smooth-pass",
        dest="multi_smooth_passes",
        action="append",
        type=multi.parse_smoothing_pass,
        metavar="OP:SECONDS",
        help="Additional smoothing pass applied to the multi-GPU column (repeatable)",
    )
    parser.add_argument(
        "--timeshare-fanouts",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Fan-out levels to include in the time-share column (default: %(default)s)",
    )
    parser.add_argument(
        "--timeshare-modes",
        type=str,
        nargs="+",
        default=["exclusive", "collocate"],
        help="Execution modes to include in the time-share column (default: %(default)s)",
    )
    parser.add_argument(
        "--timeshare-gpu-count",
        type=int,
        default=1,
        help="GPU count label to inspect for the time-share column (default: %(default)s)",
    )
    parser.add_argument(
        "--timeshare-min-activity",
        type=float,
        default=1.0,
        help="Skip time-share traces whose max utilisation stays below this threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--timeshare-smooth-window",
        type=float,
        default=timeshare.SMOOTH_WINDOW,
        help="Base smoothing window (seconds) for the time-share column (default: %(default)s)",
    )
    parser.add_argument(
        "--timeshare-smooth-operation",
        type=str,
        choices=["max", "mean"],
        default=timeshare.SMOOTH_OPERATION,
        help="Base smoothing operation for the time-share column (default: %(default)s)",
    )
    parser.add_argument(
        "--timeshare-smooth-pass",
        dest="timeshare_smooth_passes",
        action="append",
        type=multi.parse_smoothing_pass,
        metavar="OP:SECONDS",
        help="Additional smoothing pass applied to the time-share column (repeatable)",
    )
    parser.add_argument(
        "--timeshare-median-override",
        dest="timeshare_median_overrides",
        action="append",
        type=parse_median_override,
        metavar="FANOUT:MODE:RUN",
        help="Force the median selection for a specific fan-out/mode to a run index (can repeat)",
    )
    parser.add_argument(
        "--disable-default-median-overrides",
        action="store_true",
        help="Ignore the default median override map defined in plot_gpu_timeshare.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir or DEFAULT_OUTPUT
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    multi_color_map = multi.load_gpu_color_map(args.multi_color_config)
    multi_passes: Sequence[Tuple[str, float]] = (
        args.multi_smooth_passes if args.multi_smooth_passes is not None else list(multi.DEFAULT_SMOOTH_PASSES)
    )
    timeshare_passes: Sequence[Tuple[str, float]] = (
        args.timeshare_smooth_passes if args.timeshare_smooth_passes is not None else list(timeshare.SMOOTH_PASSES)
    )

    manifest_path = resolve_manifest(args.base_dir, args.manifest_path)
    run_status_map = multi.load_run_statuses(manifest_path)
    if manifest_path is None:
        print("[multi_gpu_panel] No multi_run_manifest*.jsonl detected; cannot skip in-flight runs.")
    elif run_status_map:
        print(f"[multi_gpu_panel] Loaded {len(run_status_map)} run statuses from {manifest_path}.")
    else:
        print(f"[multi_gpu_panel] Manifest {manifest_path} had no usable entries; status filtering disabled.")

    multi_fanouts = unique_preserve_order(args.multi_fanouts)
    timeshare_fanouts = unique_preserve_order(args.timeshare_fanouts)

    multi_map = multi.find_fastest_runs(
        base_dir=args.base_dir,
        fanout_levels=multi_fanouts,
        gpu_counts=args.multi_gpu_counts,
        modes=args.multi_modes,
        min_activity=args.multi_min_activity,
        run_status_map=run_status_map,
    )
    timeshare_map = collect_timeshare_entries(
        base_dir=args.base_dir,
        fanouts=timeshare_fanouts,
        modes=args.timeshare_modes,
        gpu_count=args.timeshare_gpu_count,
    )

    if not multi_map and not timeshare_map:
        print("No matching runs found across either column; nothing to render.")
        return

    if not multi_map:
        print("Warning: no multi-GPU runs found; the right column will be empty.")
    if not timeshare_map:
        print("Warning: no time-share runs found; the left column will be empty.")

    row_count = max(len(timeshare_fanouts), len(multi_fanouts))
    if row_count == 0:
        print("No fan-out levels were requested; nothing to render.")
        return

    fig_height = max(1.6 * row_count, 2.6)
    fig, axes = plt.subplots(row_count, 2, figsize=(13.0, fig_height))
    axes_rows: List[Sequence[plt.Axes]]
    if row_count == 1:
        axes_rows = [axes]
    else:
        axes_rows = list(axes)

    multi_handles: Dict[str, Line2D] = OrderedDict()
    timeshare_handles: TimeshareLegendHandles = OrderedDict()

    if args.timeshare_median_overrides:
        median_override_map: Dict[Tuple[int, str], int] = {}
        if not args.disable_default_median_overrides:
            median_override_map.update(timeshare.MEDIAN_OVERRIDES)
        for fanout, mode_key, run_idx in args.timeshare_median_overrides:
            median_override_map[(fanout, mode_key)] = run_idx
    else:
        median_override_map = {} if args.disable_default_median_overrides else dict(timeshare.MEDIAN_OVERRIDES)

    median_overrides = median_override_map if median_override_map else None

    for row_idx in range(row_count):
        row_axes = axes_rows[row_idx]
        ts_ax, multi_ax = row_axes
        ts_fanout = timeshare_fanouts[row_idx] if row_idx < len(timeshare_fanouts) else None
        multi_fanout = multi_fanouts[row_idx] if row_idx < len(multi_fanouts) else None

        ts_visible = False
        if ts_fanout is not None:
            ts_entries = timeshare_map.get(ts_fanout, [])
            ts_visible = render_timeshare_subplot(
                ts_ax,
                fanout=ts_fanout,
                entries=ts_entries,
                smooth_window=args.timeshare_smooth_window,
                smooth_operation=args.timeshare_smooth_operation,
                smooth_passes=timeshare_passes,
                min_activity=args.timeshare_min_activity,
                median_overrides=median_overrides,
                legend_handles=timeshare_handles,
            )
            if ts_visible:
                if row_idx not in (0, row_count - 1):
                    ts_ax.tick_params(labelbottom=False)
            else:
                ts_ax.cla()
                ts_ax.axis("off")
                ts_ax.text(
                    0.5,
                    0.5,
                    f"No data\nfan-out {ts_fanout}",
                    ha="center",
                    va="center",
                    transform=ts_ax.transAxes,
                    fontsize=10,
                )
        else:
            ts_ax.set_visible(False)

        multi_visible = False
        if multi_fanout is not None:
            runs = multi_map.get(multi_fanout, [])
            multi_visible = render_multi_subplot(
                multi_ax,
                fanout=multi_fanout,
                runs=runs,
                smooth_window=args.multi_smooth_window,
                smooth_operation=args.multi_smooth_operation,
                smooth_passes=multi_passes,
                color_map=multi_color_map,
                legend_handles=multi_handles,
            )
            if multi_visible:
                if row_idx not in (0, row_count - 1):
                    multi_ax.tick_params(labelbottom=False)
            else:
                multi_ax.cla()
                multi_ax.axis("off")
                multi_ax.text(
                    0.5,
                    0.5,
                    f"No data\nfan-out {multi_fanout}",
                    ha="center",
                    va="center",
                    transform=multi_ax.transAxes,
                    fontsize=10,
                )
        else:
            multi_ax.set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.7, wspace=0.08)

    figure_path = output_dir / args.figure_name
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    if figure_path.exists():
        rel_panel = os.path.relpath(figure_path, Path.cwd())
        print(f"Wrote combined panel to {rel_panel}")
    else:
        print(f"Wrote combined panel to {figure_path}")

    multi_legend_path = output_dir / MULTI_LEGEND_NAME
    if multi_handles:
        legend_path = multi.save_gpu_legend(multi_handles, multi_legend_path)
        if legend_path:
            rel_multi_legend = os.path.relpath(legend_path, Path.cwd())
            print(f"Wrote multi-GPU legend to {rel_multi_legend}")
    else:
        print("No multi-GPU legend entries were generated.")

    timeshare_legend_path = output_dir / TIMESHARE_LEGEND_NAME
    ts_result = save_timeshare_legend(timeshare_handles, timeshare_legend_path)
    if ts_result:
        rel_ts_legend = os.path.relpath(ts_result, Path.cwd())
        print(f"Wrote time-share legend to {rel_ts_legend}")
    else:
        print("No time-share legend entries were generated.")

    reduction_summaries = summarize_multi_gpu_reduction(multi_map)
    if reduction_summaries:
        print("Multi-GPU duration reduction vs GPU1 baselines:")
        for summary in reduction_summaries:
            print(f"  {summary}")


if __name__ == "__main__":
    main()
