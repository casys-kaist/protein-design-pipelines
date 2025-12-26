#!/usr/bin/env python3
"""Parse Nextflow trace files recorded in the multi-run manifest and emit summaries."""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

# ========= CONFIG =========
BASE_INPUT = "denovo_1n0r_3prs"
RESULTS_ROOT = Path("results/denovo") / BASE_INPUT
MANIFEST_PATH = RESULTS_ROOT / "multi_run_manifest.jsonl"
SUMMARY_DIR = RESULTS_ROOT / "summaries"
# ==========================


def parse_duration(duration_str: str) -> float:
    total = 0.0
    for part in duration_str.split():
        if part.endswith("h"):
            total += float(part[:-1]) * 3600
        elif part.endswith("ms"):
            total += float(part[:-2]) / 1000.0
        elif part.endswith("m"):
            total += float(part[:-1]) * 60
        elif part.endswith("s"):
            total += float(part[:-1])
    return total


def load_manifest_entries() -> Iterable[Dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    with MANIFEST_PATH.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def resolve_trace_file(entry: Dict) -> Optional[Path]:
    if entry.get("trace_file"):
        trace_path = Path(entry["trace_file"])
        if trace_path.exists():
            return trace_path

    run_outdir = Path(entry["outdir"])
    trace_dir = run_outdir / "pipeline_info"
    candidates = sorted(trace_dir.glob("execution_trace_*.txt"))
    return candidates[-1] if candidates else None


def collect_run_metrics(entries: Iterable[Dict]) -> pd.DataFrame:
    rows: List[Dict] = []

    for entry in entries:
        if entry.get("status") != "success":
            continue

        trace_path = resolve_trace_file(entry)
        if not trace_path:
            print(
                f"Warning: missing trace file for {entry['k8s_profile']} "
                f"{entry['param_set']} run {entry['run_index']}"
            )
            continue

        trace = pd.read_csv(trace_path, sep="\t")
        trace["duration_sec"] = trace["duration"].apply(parse_duration)

        grouped = trace.loc[trace.groupby("name")["duration_sec"].idxmax()].reset_index(drop=True)

        for _, task_row in grouped.iterrows():
            rows.append(
                {
                    "K8S_PROFILE": entry["k8s_profile"],
                    "PARAM_SET": ",".join(str(v) for v in entry["param_set"]),
                    "RUN": entry["run_index"],
                    "TASK": task_row["name"],
                    "DURATION_SEC": task_row["duration_sec"],
                }
            )

    return pd.DataFrame(rows)


def summarise_runs(df_runs: pd.DataFrame) -> pd.DataFrame:
    summary_rows: List[Dict] = []

    grouped = df_runs.groupby(["K8S_PROFILE", "PARAM_SET", "TASK"], as_index=False)
    for (profile, param_label, task), group in grouped:
        group_sorted = group.sort_values("DURATION_SEC")
        trimmed = group_sorted.iloc[1:-1] if len(group_sorted) > 2 else group_sorted
        summary_rows.append(
            {
                "K8S_PROFILE": profile,
                "PARAM_SET": param_label,
                "TASK": task,
                "AVG_DURATION_SEC": trimmed["DURATION_SEC"].mean(),
                "NUM_RUNS": len(group),
            }
        )

    return pd.DataFrame(summary_rows)


def write_outputs(df_runs: pd.DataFrame, df_summary: pd.DataFrame) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    runs_csv = SUMMARY_DIR / "run_metrics.csv"
    df_runs.to_csv(runs_csv, index=False)
    print(f"Wrote per-run metrics to {runs_csv}")

    summary_csv = SUMMARY_DIR / "task_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"Wrote task summary to {summary_csv}")

    for (profile, param_label), group in df_summary.groupby(["K8S_PROFILE", "PARAM_SET"]):
        filename_fragment = param_label.replace(",", "_")
        profile_fragment = profile.replace("/", "_")
        out_csv = SUMMARY_DIR / f"profile_result_collocation_{filename_fragment}_{profile_fragment}.csv"
        group.to_csv(out_csv, index=False)
        print(f"Wrote summary to {out_csv}")


def main() -> None:
    entries = list(load_manifest_entries())
    df_runs = collect_run_metrics(entries)
    if df_runs.empty:
        print("No successful runs found in manifest.")
        return

    df_summary = summarise_runs(df_runs)
    write_outputs(df_runs, df_summary)


if __name__ == "__main__":
    main()
