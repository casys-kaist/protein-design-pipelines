#!/usr/bin/env python3
"""
Generate profiling summary graphs as a single long figure with 4 subplots (1 row).

Changes vs runtime/analysis version:
- No 3x3 faceting; each subplot is a single bar chart across components.
- No x-label rotation (kept horizontal).
- Color by component; encode quality (low/medium/high) via hatch patterns.
- Single shared legend area across all 4 subplots.
- Default output directory moved to profiling/plots/outputs.

Inputs layout (produced by profiling pipeline):

PROFILE_DATA_ROOT (default: /mnt/nfs/new/bioinformatics/profile/{raw,results,parsed_csv})
  <component>/
    level_{a,b,c}/<sample_id>/<quality>/run_N/*.csv  # new Hydra layout (preferred)
    <timestamp>/<sample_id>/<quality>/run_N/*.csv    # results snapshots
    <scaffold>_<ligand>/<quality>/*utilization.csv   # legacy parsed_csv fallback

Outputs:
  - profiling/plots/outputs/summary_metrics.csv
  - profiling/plots/outputs/metrics_overview.png

Usage:
  python scripts/plot_microbench.py \
    --data-root /mnt/nfs/new/bioinformatics/profile/parsed_csv \
    --experiments profiling/configs/experiments/minimal.yaml \
    --outdir profiling/plots/outputs
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
try:
    import yaml  # optional
except Exception:
    yaml = None
from matplotlib import font_manager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure the profiling package (profiling/src/profile) wins over the stdlib "profile".
PROFILING_SRC = REPO_ROOT / "profiling" / "src"
if PROFILING_SRC.exists() and str(PROFILING_SRC) not in sys.path:
    sys.path.insert(0, str(PROFILING_SRC))

try:
    sys.modules.pop("profile", None)
    from profile import DATA_ROOT, RAW_DATA_ROOT, RESULTS_ROOT
except Exception:
    DATA_ROOT = Path("/mnt/nfs/new/bioinformatics/profile/results")
    RAW_DATA_ROOT = Path("/mnt/nfs/new/bioinformatics/profile/raw")
    RESULTS_ROOT = DATA_ROOT

# Allow plotting script to walk multiple roots (modern parsed CSV + legacy snapshots)
DEFAULT_DATA_ROOTS: List[Path] = [
    RAW_DATA_ROOT,
]

QUALITY_ORDER = ["low", "medium", "high"]
LIGAND_ORDER = ["3dx1", "4de1", "3prs"]
TIMESTAMP_DIR_RE = re.compile(r"^\d{8}_\d{6}$")
UTIL_SUFFIX = "_utilization.csv"
NCU_SUFFIX = "_ncu_summary.csv"
KERNEL_SUFFIX = "_kernel_summary.csv"
TELEMETRY_SUFFIX = "_telemetry.csv"


def _looks_like_timestamp(name: str) -> bool:
    return bool(TIMESTAMP_DIR_RE.fullmatch(name.strip()))


def _safe_mtime(path: Optional[Path]) -> float:
    if path is None:
        return float("-inf")
    try:
        return path.stat().st_mtime
    except OSError:
        return float("-inf")


def _pick_latest_path(paths: Iterable[Path]) -> Optional[Path]:
    best_path: Optional[Path] = None
    best_mtime = float("-inf")
    for path in paths:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime >= best_mtime:
            best_mtime = mtime
            best_path = path
    return best_path


def _resolve_partner_file(util_csv: Path, target_suffix: str) -> Optional[Path]:
    name = util_csv.name
    if name.endswith(UTIL_SUFFIX):
        partner_name = name[: -len(UTIL_SUFFIX)] + target_suffix
        candidate = util_csv.with_name(partner_name)
        if candidate.exists():
            return candidate
    matches = list(util_csv.parent.glob(f"*{target_suffix}"))
    return _pick_latest_path(matches)


def _canon_component_name(name: str) -> str:
    return re.sub(r"[^a-z]", "", str(name).lower())


def _is_color_like(value: str) -> bool:
    if not isinstance(value, str):
        return False
    val = value.strip()
    if not val:
        return False
    if val.startswith("#"):
        body = val[1:]
        return bool(re.fullmatch(r"[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8}", body))
    lowered = val.lower()
    if lowered.startswith(("rgb(", "rgba(", "hsl(", "hsla(", "cmyk(", "lab(")):
        return True
    if lowered.startswith("tab:"):
        return True
    if re.fullmatch(r"c\d+", lowered):
        return True
    if lowered in {"black", "white", "red", "blue", "green", "yellow", "cyan", "magenta", "orange", "purple", "brown", "grey", "gray"}:
        return True
    return False


def _extract_color_value(entry) -> Optional[str]:
    if isinstance(entry, str) and _is_color_like(entry):
        return entry.strip()
    if isinstance(entry, dict):
        for key in ("color", "colour", "hex", "value", "code", "rgb"):
            val = entry.get(key)
            if isinstance(val, str) and _is_color_like(val):
                return val.strip()
    return None


def _normalize_component_dict(raw) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    normalized: Dict[str, str] = {}
    for key, value in raw.items():
        color = _extract_color_value(value)
        if color:
            normalized[str(key)] = color
    return normalized


def _merge_aliases(target: Dict[str, str], raw) -> None:
    if isinstance(raw, dict):
        for alias, mapped in raw.items():
            if isinstance(alias, str):
                if isinstance(mapped, str):
                    target[str(alias)] = mapped
                elif isinstance(mapped, (list, tuple)):
                    for item in mapped:
                        if isinstance(item, str):
                            target[str(alias)] = item
                            break


def _parse_palette_entry(entry, alias_map: Dict[str, str]) -> Dict[str, str]:
    if isinstance(entry, dict):
        for alias_key in ("aliases", "alias", "component_aliases", "component_alias", "alias_map"):
            _merge_aliases(alias_map, entry.get(alias_key))
        if "components" in entry:
            colors = _normalize_component_dict(entry["components"])
            if colors:
                return colors
        if "colors" in entry:
            colors = _normalize_component_dict(entry["colors"])
            if colors:
                return colors
        if "palette" in entry:
            colors = _normalize_component_dict(entry["palette"])
            if colors:
                return colors
        return _normalize_component_dict(entry)
    if isinstance(entry, list):
        combined: Dict[str, str] = {}
        for item in entry:
            if isinstance(item, dict):
                combined.update(_normalize_component_dict(item))
        return combined
    return {}


def _extract_palette_from_config(data) -> Tuple[Dict[str, str], Dict[str, str]]:
    alias_map: Dict[str, str] = {}
    if not isinstance(data, (dict, list)):
        return {}, alias_map

    def _pick_from_palettes(palettes) -> Dict[str, str]:
        palette_name: Optional[str] = None
        name_keys = (
            "current_palette",
            "default_palette",
            "active_palette",
            "selected_palette",
            "palette",
            "default",
            "current",
            "active",
            "selected",
        )
        for key in name_keys:
            val = None
            if isinstance(data, dict):
                val = data.get(key)
            if isinstance(val, str) and val:
                palette_name = val
                break
        if isinstance(palettes, dict):
            if palette_name:
                entry = palettes.get(palette_name)
                if entry is None:
                    lookup = {
                        _canon_component_name(k): k
                        for k in palettes.keys()
                        if isinstance(k, str)
                    }
                    entry = palettes.get(lookup.get(_canon_component_name(palette_name)))
                if entry is not None:
                    colors = _parse_palette_entry(entry, alias_map)
                    if colors:
                        return colors
            for entry in palettes.values():
                colors = _parse_palette_entry(entry, alias_map)
                if colors:
                    return colors
        elif isinstance(palettes, list):
            if palette_name:
                for entry in palettes:
                    if not isinstance(entry, dict):
                        continue
                    name = entry.get("name") or entry.get("id")
                    if isinstance(name, str) and _canon_component_name(name) == _canon_component_name(palette_name):
                        colors = _parse_palette_entry(entry, alias_map)
                        if colors:
                            return colors
            for entry in palettes:
                colors = _parse_palette_entry(entry, alias_map)
                if colors:
                    return colors
        return {}

    queue: list = []
    if isinstance(data, dict):
        for alias_key in ("aliases", "alias", "component_aliases", "component_alias", "alias_map"):
            _merge_aliases(alias_map, data.get(alias_key))
        for key in ("components", "component_colors", "component_colours", "colors", "colours"):
            if key in data:
                colors = _normalize_component_dict(data[key])
                if colors:
                    return colors, alias_map
        for key in ("palette", "scheme", "set"):
            if key in data:
                colors = _parse_palette_entry(data[key], alias_map)
                if colors:
                    return colors, alias_map
        if "palettes" in data:
            colors = _pick_from_palettes(data["palettes"])
            if colors:
                return colors, alias_map
        queue.append(data)
    elif isinstance(data, list):
        queue.extend(data)

    visited: set[int] = set()
    while queue:
        node = queue.pop(0)
        if id(node) in visited:
            continue
        visited.add(id(node))
        if isinstance(node, dict):
            for alias_key in ("aliases", "alias", "component_aliases", "component_alias", "alias_map"):
                _merge_aliases(alias_map, node.get(alias_key))
            colors = _normalize_component_dict(node)
            if colors:
                return colors, alias_map
            for value in node.values():
                if isinstance(value, (dict, list)):
                    queue.append(value)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    queue.append(item)
    return {}, alias_map


def _resolve_component_colors(
    comp_order: List[str],
    raw_colors: Dict[str, str],
    alias_map: Dict[str, str],
) -> Dict[str, str]:
    if not raw_colors:
        return {}
    resolved: Dict[str, str] = {}
    canon_color_map = {_canon_component_name(key): val for key, val in raw_colors.items()}
    direct_alias_colors: Dict[str, str] = {}
    alias_lookup: Dict[str, str] = {}
    for alias, target in alias_map.items():
        if not isinstance(alias, str):
            continue
        canon_alias = _canon_component_name(alias)
        if isinstance(target, str):
            if _is_color_like(target):
                direct_alias_colors[canon_alias] = target.strip()
            else:
                alias_lookup[canon_alias] = _canon_component_name(target)
        elif isinstance(target, (list, tuple)):
            for item in target:
                if isinstance(item, str):
                    if _is_color_like(item):
                        direct_alias_colors[canon_alias] = item.strip()
                        break
                    alias_lookup[canon_alias] = _canon_component_name(item)
                    if alias_lookup[canon_alias] in canon_color_map:
                        break
    for comp in comp_order:
        if comp in raw_colors:
            resolved[comp] = raw_colors[comp]
            continue
        canon_name = _canon_component_name(comp)
        if canon_name in direct_alias_colors:
            resolved[comp] = direct_alias_colors[canon_name]
            continue
        if canon_name in canon_color_map:
            resolved[comp] = canon_color_map[canon_name]
            continue
        mapped = alias_lookup.get(canon_name)
        if mapped:
            if mapped in canon_color_map:
                resolved[comp] = canon_color_map[mapped]
            elif mapped in direct_alias_colors:
                resolved[comp] = direct_alias_colors[mapped]
    return resolved


def _parse_simple_color_mapping(text: str) -> Dict[str, str]:
    simple: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        hash_index = value.find("#")
        if hash_index > 0 and value[hash_index - 1].isspace():
            value = value[:hash_index].strip()
        if not key or not value:
            continue
        if (value.startswith("\"") and value.endswith("\"")) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1].strip()
        color = _extract_color_value(value)
        if color:
            simple[key] = color
    return simple


def _load_color_config_from_path(path: Path):
    try:
        text = path.read_text()
    except Exception:
        return None
    if yaml is not None:
        try:
            loaded = yaml.safe_load(text)
            if isinstance(loaded, (dict, list)):
                return loaded
        except Exception:
            pass
    return _parse_simple_color_mapping(text)


def load_component_color_map(comp_order: List[str]) -> Dict[str, str]:
    color_sources: List[Path] = [
        SCRIPT_DIR / "colors.yaml",
        SCRIPT_DIR.parent / "colors.yaml",
        Path.cwd() / "colors.yaml",
        REPO_ROOT / "profiling" / "plots" / "colors.yaml",
    ]
    seen: set[Path] = set()
    for path in color_sources:
        if not path.exists():
            continue
        try:
            resolved_path = path.resolve()
        except Exception:
            resolved_path = path
        if resolved_path in seen:
            continue
        seen.add(resolved_path)
        loaded = _load_color_config_from_path(path)
        if not loaded:
            continue
        palette, aliases = _extract_palette_from_config(loaded)
        if not palette:
            continue
        resolved = _resolve_component_colors(comp_order, palette, aliases)
        if resolved:
            return resolved
    return {}


def _normalize_quality_label(q: str) -> str:
    ql = str(q).strip().lower()
    if ql in ("low", "l"):
        return "low"
    if ql in ("medium", "med", "mid", "m"):
        return "medium"
    if ql in ("high", "hi", "h"):
        return "high"
    return ql


def normalize_sample_id(sample_id: str) -> str:
    if not sample_id:
        return sample_id
    return str(sample_id)


def resolve_experiments_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    raw = Path(value)
    candidates: List[Path] = [raw]
    if not raw.suffix:
        candidates.append(raw.with_suffix(".yaml"))
    for base in (REPO_ROOT / "profiling" / "configs" / "experiments", SCRIPT_DIR / "experiments"):
        candidates.append(base / raw)
        if not raw.suffix:
            candidates.append(base / raw.with_suffix(".yaml"))
    seen: Set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            return path
    return None


def load_experiment_filters(path_str: Optional[str]) -> Dict[str, Dict[str, Set[str]]]:
    if not path_str:
        return {}
    experiments_path = resolve_experiments_path(path_str)
    if experiments_path is None:
        print(f"[WARN] experiments file not found: {path_str}", file=sys.stderr)
        return {}
    if yaml is None:
        raise SystemExit("PyYAML is required to parse experiments YAML (install pyyaml).")
    try:
        data = yaml.safe_load(experiments_path.read_text()) or {}
    except Exception as exc:
        print(f"[WARN] failed to read experiments file {experiments_path}: {exc}", file=sys.stderr)
        return {}
    exp_data = data.get("experiments") if isinstance(data, dict) else None
    if exp_data is None and isinstance(data, dict):
        exp_data = data
    if not isinstance(exp_data, dict):
        return {}
    filters: Dict[str, Dict[str, Set[str]]] = {}
    for comp, cfg in exp_data.items():
        if not isinstance(cfg, dict):
            continue
        samples = cfg.get("sample_id") or cfg.get("sample_ids") or cfg.get("samples") or []
        if isinstance(samples, str):
            samples = [samples]
        sample_set: Set[str] = {str(s).lower() for s in samples if s}
        qualities = cfg.get("quality") or cfg.get("qualities") or []
        if isinstance(qualities, str):
            qualities = [qualities]
        qual_set: Set[str] = {_normalize_quality_label(q) for q in qualities if q}
        filters[_canon_component_name(comp)] = {"samples": sample_set, "qualities": qual_set}
    return filters


def _sample_matches(sample_id: str, allowed: Set[str]) -> bool:
    if not allowed:
        return True
    text = str(sample_id).lower()
    for item in allowed:
        if not item:
            continue
        target = str(item).lower()
        if text == target or text.endswith(target) or target in text:
            return True
    return False


def filter_records_by_experiments(df: pd.DataFrame, filters: Dict[str, Dict[str, Set[str]]]) -> pd.DataFrame:
    if not filters or df.empty:
        return df

    def _keep(row: pd.Series) -> bool:
        comp_key = _canon_component_name(row.get("component", ""))
        cfg = filters.get(comp_key)
        if cfg is None:
            return False
        sample_val = str(row.get("sample_id") or "").lower()
        if not _sample_matches(sample_val, cfg.get("samples", set())):
            return False
        q_val = _normalize_quality_label(row.get("quality", ""))
        qualities = cfg.get("qualities", set())
        if qualities and q_val not in qualities:
            return False
        return True

    mask = df.apply(_keep, axis=1)
    return df[mask].reset_index(drop=True)


def find_gpu_columns(df: pd.DataFrame, suffixes) -> List[str]:
    if isinstance(suffixes, str):
        suffixes = [suffixes]
    cols: List[str] = []
    for raw_col in df.columns:
        col = str(raw_col)
        for suffix in suffixes:
            pat = re.compile(rf"^gpu\d+_{re.escape(str(suffix))}$", re.IGNORECASE)
            if pat.match(col):
                cols.append(raw_col)
                break
    return cols


def compute_wall_clock(util_df: pd.DataFrame) -> float:
    if util_df is None or util_df.empty:
        return float("nan")
    if "elapsed_s" in util_df.columns:
        try:
            return float(pd.to_numeric(util_df["elapsed_s"], errors="coerce").max())
        except Exception:
            pass
    # Sometimes elapsed is encoded in milliseconds or other numeric columns
    for col in util_df.columns:
        cl = str(col).lower()
        if not cl.startswith("elapsed"):
            continue
        series = pd.to_numeric(util_df[col], errors="coerce")
        if series.notna().any():
            if cl.endswith("ms"):
                return float(series.max() / 1000.0)
            return float(series.max())
    if "timestamp" in util_df.columns:
        try:
            ts = pd.to_datetime(util_df["timestamp"], errors="coerce", format="ISO8601")
        except Exception:
            ts = pd.to_datetime(util_df["timestamp"], errors="coerce")
        ts = ts.dropna()
        if len(ts) >= 2:
            delta = ts.max() - ts.min()
            try:
                return float(delta.total_seconds())
            except Exception:
                return float("nan")
    return float("nan")


def compute_temporal_util(util_df: pd.DataFrame) -> float:
    """
    Temporal utilization approximated as the time-averaged SM busy percent
    (or GPU utilization when SM busy is unavailable) of the most-active GPU at each sample.
    """
    if util_df is None or util_df.empty:
        return float("nan")

    def _mean_of_cols(cols: List[str]) -> float:
        if not cols:
            return float("nan")
        vals = util_df[cols].apply(pd.to_numeric, errors="coerce")
        row_max = vals.max(axis=1)
        return float(row_max.mean())

    sm_cols = find_gpu_columns(util_df, ["sm_percent", "sm_util", "sm"])
    if sm_cols:
        val = _mean_of_cols(sm_cols)
        if not np.isnan(val):
            return val

    util_cols = find_gpu_columns(util_df, "util")
    return _mean_of_cols(util_cols)


def compute_peak_memory(util_df: pd.DataFrame) -> float:
    mem_cols = find_gpu_columns(util_df, ["vram_used_MiB", "mem_used", "memory_used"])
    if not mem_cols:
        return float("nan")
    vals = util_df[mem_cols].apply(pd.to_numeric, errors="coerce")
    return float(vals.max(axis=1).max())


def parse_level_a_telemetry_csv(path: Path) -> Tuple[float, float, float]:
    wall_clock_s = float("nan")
    temporal_util = float("nan")
    peak_mem = float("nan")
    try:
        rows = list(csv.reader(Path(path).open()))
    except Exception:
        return wall_clock_s, temporal_util, peak_mem
    if len(rows) <= 1:
        return wall_clock_s, temporal_util, peak_mem

    timestamps: List[pd.Timestamp] = []
    per_row_util: List[float] = []
    per_row_mem: List[float] = []

    for row in rows[1:]:
        if len(row) < 4:
            continue
        ts = pd.to_datetime(row[0], errors="coerce")
        if pd.notna(ts):
            timestamps.append(ts)

        tokens = row[3:]
        if not tokens:
            continue
        chunk = None
        if len(tokens) % 6 == 0:
            chunk = 6
        elif len(tokens) % 5 == 0:
            chunk = 5  # legacy header without GPU index
        if chunk is None:
            continue

        max_util = float("nan")
        max_mem = float("nan")
        for i in range(0, len(tokens), chunk):
            chunk_vals = tokens[i : i + chunk]
            if len(chunk_vals) < chunk:
                break
            try:
                util_val = float(chunk_vals[2 if chunk == 6 else 1])
            except Exception:
                util_val = float("nan")
            try:
                mem_used = float(chunk_vals[5 if chunk == 6 else 4])
            except Exception:
                mem_used = float("nan")
            if pd.notna(util_val):
                max_util = max(util_val, max_util) if pd.notna(max_util) else util_val
            if pd.notna(mem_used):
                max_mem = max(mem_used, max_mem) if pd.notna(max_mem) else mem_used
        if pd.notna(max_util):
            per_row_util.append(max_util)
        if pd.notna(max_mem):
            per_row_mem.append(max_mem)

    if timestamps:
        wall_clock_s = float((max(timestamps) - min(timestamps)).total_seconds())
    if per_row_util:
        temporal_util = float(pd.Series(per_row_util).mean())
    if per_row_mem:
        peak_mem = float(pd.Series(per_row_mem).max())
    return wall_clock_s, temporal_util, peak_mem


def load_kernel_time_weights(kernel_summary_csv: Path) -> Optional[pd.DataFrame]:
    if not kernel_summary_csv.exists():
        return None
    df = pd.read_csv(kernel_summary_csv)
    # Expect columns: Index,Time (%),Name
    time_col = None
    name_col = None
    for c in df.columns:
        cl = c.strip().lower()
        if cl.startswith("time"):
            time_col = c
        if cl in ("name", "kernel name"):
            name_col = c
    if time_col is None or name_col is None:
        return None
    agg = df.groupby(name_col, as_index=False)[time_col].sum()
    agg.rename(columns={name_col: "kernel_name", time_col: "time_pct"}, inplace=True)
    total = agg["time_pct"].sum()
    agg["time_w"] = (agg["time_pct"] / total) if total > 0 else 0.0
    return agg


def load_ncu_achieved_occupancy(ncu_summary_csv: Path) -> Optional[pd.DataFrame]:
    if not ncu_summary_csv.exists():
        return None
    df = pd.read_csv(ncu_summary_csv)
    kname = None
    ach = None
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("kernel name", "name"):
            kname = c
        if "achieved" in cl and "occup" in cl:
            ach = c
    if kname is None or ach is None:
        return None
    occ = df.groupby(kname, as_index=False)[ach].mean()
    occ.rename(columns={kname: "kernel_name", ach: "achieved_occupancy"}, inplace=True)
    return occ


def compute_spatial_util(kernel_summary_csv: Path, ncu_summary_csv: Path) -> float:
    kws = load_kernel_time_weights(kernel_summary_csv)
    occ = load_ncu_achieved_occupancy(ncu_summary_csv)
    if kws is None or occ is None or kws.empty or occ.empty:
        return float("nan")
    merged = pd.merge(kws, occ, on="kernel_name", how="inner")
    if merged.empty:
        return float("nan")
    val = (merged["achieved_occupancy"] * merged["time_w"]).sum()
    return float(val)


def compute_memory_util_from_nsys(nsys_csv: Path) -> float:
    """Best-effort extraction of device memory utilization (%) from an NSYS CSV."""
    try:
        df = pd.read_csv(nsys_csv)
    except Exception:
        return float("nan")
    if df is None or df.empty:
        return float("nan")
    def norm(s: str) -> str:
        return str(s).strip().lower()
    # Column-based pattern search
    for c in df.columns:
        cl = norm(c)
        if ("mem" in cl or "dram" in cl) and ("util" in cl or "pct" in cl or "percent" in cl or "utilization" in cl):
            series = pd.to_numeric(df[c], errors="coerce").dropna()
            if not series.empty:
                return float(series.mean())
        if "device memory utilization" in cl:
            series = pd.to_numeric(df[c], errors="coerce").dropna()
            if not series.empty:
                return float(series.mean())
    # Name/Value table style
    name_cols = [c for c in df.columns if norm(c) in ("metric", "metric name", "name")]
    val_cols = [c for c in df.columns if any(k in norm(c) for k in ("value", "avg", "mean", "percent", "util"))]
    if name_cols and val_cols:
        ncol, vcol = name_cols[0], val_cols[0]
        mask = df[ncol].astype(str).str.lower().str.contains("memory") & (
            df[ncol].astype(str).str.lower().str.contains("util") | df[ncol].astype(str).str.lower().str.contains("dram")
        )
        series = pd.to_numeric(df.loc[mask, vcol], errors="coerce").dropna()
        if not series.empty:
            return float(series.mean())
    return float("nan")


def infer_quality_from_name(name: str, default: str = "medium") -> str:
    lowered = str(name).lower()
    for q in QUALITY_ORDER:
        if q in lowered:
            return q
    if "throughput" in lowered:
        return "low"
    if "accuracy" in lowered:
        return "high"
    return default


def _progress_iter(iterable, desc: str, enable: bool = True):
    if enable and tqdm is not None:
        return tqdm(iterable, desc=desc, leave=False)
    return iterable


def _collect_records_from_root(data_root: Path, *, verbose: bool = False, progress: bool = True) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    telemetry_re = re.compile(
        r"^(?P<comp>[a-z0-9]+)[-_](?P<sample>[A-Za-z0-9_]+?)[-_](?P<quality>low|medium|high)(?:[-_]r?\d+)?_telemetry\\.csv$",
        re.IGNORECASE,
    )
    ncu_re = re.compile(
        r"^(?P<comp>[a-z0-9]+)[-_](?P<sample>[A-Za-z0-9_]+?)[-_](?P<quality>low|medium|high)[-_]r?\d+_ncu_summary\\.csv$",
        re.IGNORECASE,
    )
    ksum_re = re.compile(
        r"^(?P<comp>[a-z0-9]+)[-_](?P<sample>[A-Za-z0-9_]+?)[-_](?P<quality>low|medium|high)[-_]r?\d+_kernel_summary\\.csv$",
        re.IGNORECASE,
    )

    def infer_profiler_level(path: Path, hint: Optional[str] = None) -> str:
        if hint:
            return hint
        lower_parts = [part.lower() for part in path.parts]
        for lvl in ("level_c", "level_b", "level_a"):
            if any(lvl in part for part in lower_parts):
                return lvl
        return "unknown"

    def add_record(
        *,
        component: str,
        sample_id: str,
        quality: str,
        profiler_level: str,
        util_csv: Optional[Path],
        ncu_csv: Optional[Path],
        ksum_csv: Optional[Path],
        artifact_mtime: float,
    ) -> None:
        quality = _normalize_quality_label(quality)
        sample_id = normalize_sample_id(sample_id or "unknown")
        wall_clock_s = float("nan")
        temporal_util = float("nan")
        peak_mem = float("nan")
        spatial_util = float("nan")

        if util_csv and util_csv.exists():
            parsed = False
            if util_csv.name.endswith(TELEMETRY_SUFFIX):
                try:
                    wall_clock_s, temporal_util, peak_mem = parse_level_a_telemetry_csv(util_csv)
                    parsed = True
                except Exception:
                    parsed = False
            if not parsed:
                try:
                    util_df = pd.read_csv(util_csv)
                    wall_clock_s = compute_wall_clock(util_df)
                    temporal_util = compute_temporal_util(util_df)
                    peak_mem = compute_peak_memory(util_df)
                except Exception:
                    pass

        if ksum_csv and ncu_csv and ksum_csv.exists() and ncu_csv.exists():
            try:
                spatial_util = compute_spatial_util(ksum_csv, ncu_csv)
            except Exception:
                pass

        if verbose:
            print(
                "[INPUT]",
                f"root={data_root}",
                f"component={component}",
                f"sample={sample_id}/{quality}",
                f"level={profiler_level}",
                f"util={'found' if util_csv and util_csv.exists() else 'missing'}",
                f"ncu={'found' if ncu_csv and ncu_csv.exists() else 'missing'}",
                f"kernel={'found' if ksum_csv and ksum_csv.exists() else 'missing'}",
                f"util_path={util_csv}",
                f"ncu_path={ncu_csv}",
                f"kernel_path={ksum_csv}",
                f"mtime={artifact_mtime:.0f}" if artifact_mtime > 0 else "mtime=unknown",
            )

        records.append(
            {
                "component": component,
                "sample_id": sample_id,
                "quality": quality,
                "profiler_level": profiler_level or "",
                "component_config": f"{component}-{quality}",
                "wall_clock_s": wall_clock_s,
                "temporal_util_percent": temporal_util,
                "spatial_util_percent": spatial_util,
                "peak_memory_mib": peak_mem,
                "util_csv": str(util_csv) if util_csv else "",
                "ncu_csv": str(ncu_csv) if ncu_csv else "",
                "kernel_summary_csv": str(ksum_csv) if ksum_csv else "",
                "artifact_mtime": artifact_mtime,
            }
        )

    def extract_sample_id(sample_hint: str, util_csv: Optional[Path], ncu_csv: Optional[Path], ksum_csv: Optional[Path]) -> str:
        for path, regex in (
            (util_csv, telemetry_re),
            (ncu_csv, ncu_re),
            (ksum_csv, ksum_re),
        ):
            if path is None:
                continue
            match = regex.match(path.name)
            if match and match.group("sample"):
                return match.group("sample")
        return sample_hint

    def discover_artifacts(base_dir: Path):
        util_candidates: List[Path] = []
        for suffix in (UTIL_SUFFIX, TELEMETRY_SUFFIX):
            util_candidates.extend(base_dir.rglob(f"*{suffix}"))
        util_csv = _pick_latest_path(util_candidates)
        ncu_csv = _pick_latest_path(base_dir.rglob(f"*{NCU_SUFFIX}"))
        ksum_csv = _pick_latest_path(base_dir.rglob(f"*{KERNEL_SUFFIX}"))
        artifact_mtime = max(_safe_mtime(util_csv), _safe_mtime(ncu_csv), _safe_mtime(ksum_csv))
        return util_csv, ncu_csv, ksum_csv, artifact_mtime

    def iter_scaffold_dirs(comp_dir: Path):
        for child in sorted([p for p in comp_dir.iterdir() if p.is_dir()]):
            if _looks_like_timestamp(child.name):
                for nested in sorted([p for p in child.iterdir() if p.is_dir()]):
                    yield nested
            else:
                yield child

    def iter_quality_dirs(sl_dir: Path):
        q_subdirs = [p for p in sl_dir.iterdir() if p.is_dir()]
        quality_dirs: List[Tuple[Path, str]] = []
        for q_dir in q_subdirs:
            norm_q = _normalize_quality_label(q_dir.name)
            if norm_q in QUALITY_ORDER:
                quality_dirs.append((q_dir, norm_q))
        if quality_dirs:
            quality_dirs.sort(key=lambda item: QUALITY_ORDER.index(item[1]))
            return quality_dirs
        return [(sl_dir, "medium")]

    def handle_level_layout(comp_dir: Path):
        level_dirs = [p for p in comp_dir.iterdir() if p.is_dir() and p.name.lower().startswith("level_")]
        for level_dir in sorted(level_dirs):
            profiler_level = level_dir.name
            for sample_dir in sorted([p for p in level_dir.iterdir() if p.is_dir()]):
                sample_hint = sample_dir.name
                q_dirs = []
                for q_dir in sorted([p for p in sample_dir.iterdir() if p.is_dir()]):
                    q_norm = _normalize_quality_label(q_dir.name)
                    if q_norm in QUALITY_ORDER:
                        q_dirs.append((q_dir, q_norm))
                if not q_dirs:
                    q_dirs = [(sample_dir, "medium")]
                for q_dir, quality in q_dirs:
                    run_dirs = [p for p in q_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
                    targets = []
                    if run_dirs:
                        latest_run = _pick_latest_path(run_dirs)
                        if latest_run:
                            targets.append(latest_run)
                    else:
                        targets.append(q_dir)
                    for target in targets:
                        util_csv, ncu_csv, ksum_csv, artifact_mtime = discover_artifacts(target)
                        if util_csv is None and (ncu_csv is None or ksum_csv is None):
                            continue
                        sample_id = extract_sample_id(sample_hint, util_csv, ncu_csv, ksum_csv)
                        add_record(
                            component=comp_dir.name,
                            sample_id=sample_id,
                            quality=quality,
                            profiler_level=profiler_level,
                            util_csv=util_csv,
                            ncu_csv=ncu_csv,
                            ksum_csv=ksum_csv,
                            artifact_mtime=artifact_mtime,
                        )

    def handle_legacy_layout(comp_dir: Path):
        for sl_dir in iter_scaffold_dirs(comp_dir):
            sample_hint = sl_dir.name
            for q_dir, quality in iter_quality_dirs(sl_dir):
                util_csv, ncu_csv, ksum_csv, artifact_mtime = discover_artifacts(q_dir)
                if util_csv is None and (ncu_csv is None or ksum_csv is None):
                    continue
                sample_id = extract_sample_id(sample_hint, util_csv, ncu_csv, ksum_csv)
                profiler_level = infer_profiler_level(q_dir)
                add_record(
                    component=comp_dir.name,
                    sample_id=sample_id,
                    quality=quality,
                    profiler_level=profiler_level,
                    util_csv=util_csv,
                    ncu_csv=ncu_csv,
                    ksum_csv=ksum_csv,
                    artifact_mtime=artifact_mtime,
                )

    comp_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    for comp_dir in _progress_iter(comp_dirs, desc=str(data_root), enable=progress):
        component = comp_dir.name
        if component in {"README.md"}:
            continue
        level_dirs = [p for p in comp_dir.iterdir() if p.is_dir() and p.name.lower().startswith("level_")]
        if level_dirs:
            handle_level_layout(comp_dir)
        else:
            handle_legacy_layout(comp_dir)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    dedup_cols = ["component", "sample_id", "quality", "profiler_level"]
    df.sort_values("artifact_mtime", inplace=True)
    df = df.drop_duplicates(subset=dedup_cols, keep="last").reset_index(drop=True)
    return df


def collect_records(data_roots: List[Path], *, verbose: bool = False, progress: bool = True) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    seen_roots: Set[Path] = set()
    for priority, root in enumerate(data_roots):
        root_path = Path(root).expanduser()
        try:
            resolved_root = root_path.resolve()
        except Exception:
            resolved_root = root_path
        if resolved_root in seen_roots:
            continue
        seen_roots.add(resolved_root)
        if not root_path.exists():
            print(f"[WARN] data root not found, skipping: {root_path}", file=sys.stderr)
            continue
        df = _collect_records_from_root(root_path, verbose=verbose, progress=progress)
        if verbose:
            print(f"[INPUT] root={root_path} rows={len(df)}")
        if not df.empty:
            df["data_root"] = str(root_path)
            df["root_priority"] = priority
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    dedup_cols = ["component", "sample_id", "quality", "profiler_level"]
    level_priority = {"level_a": 0, "level_b": 1, "level_c": 2}
    if "profiler_level" in combined.columns:
        combined["level_rank"] = combined["profiler_level"].map(lambda x: level_priority.get(str(x).lower(), 9))
    else:
        combined["level_rank"] = 9
    if "root_priority" in combined.columns and "artifact_mtime" in combined.columns:
        combined.sort_values(
            ["root_priority", "level_rank", "artifact_mtime"],
            ascending=[True, True, False],
            inplace=True,
        )
        combined = combined.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    drop_cols = [col for col in ("artifact_mtime", "root_priority") if col in combined.columns]
    if drop_cols:
        combined = combined.drop(columns=drop_cols)
    return combined


def collapse_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine multi-level rows into a single row per (component, sample_id, quality)
    using metric-specific level preferences:
      - wall_clock_s / temporal_util_percent / peak_memory_mib: prefer level_a, then b, then c
      - spatial_util_percent: prefer level_c, then b, then a
    """
    if df.empty:
        return df
    preferences = {
        "wall_clock_s": ["level_a", "level_b", "level_c", ""],
        "temporal_util_percent": ["level_a", "level_b", "level_c", ""],
        "peak_memory_mib": ["level_a", "level_b", "level_c", ""],
        "spatial_util_percent": ["level_c", "level_b", "level_a", ""],
    }

    def _pick_value(group: pd.DataFrame, metric: str) -> float:
        order = preferences.get(metric, ["level_a", "level_b", "level_c", ""])
        for level in order:
            if level:
                subset = group[group["profiler_level"].str.lower() == level]
            else:
                subset = group
            if subset.empty or metric not in subset.columns:
                continue
            series = pd.to_numeric(subset[metric], errors="coerce")
            series = series.dropna()
            if not series.empty:
                return float(series.iloc[0])
        return float("nan")

    collapsed_rows: List[Dict[str, object]] = []
    group_cols = ["component", "sample_id", "quality"]
    for (comp, sample, qual), grp in df.groupby(group_cols, as_index=False):
        row: Dict[str, object] = {"component": comp, "sample_id": sample, "quality": qual}
        for metric in ("wall_clock_s", "temporal_util_percent", "peak_memory_mib", "spatial_util_percent"):
            row[metric] = _pick_value(grp, metric)
        # Keep artifact pointers from the preferred levels for debugging
        if "util_csv" in grp.columns:
            util_row = grp[grp["profiler_level"].str.lower() == "level_a"]
            if util_row.empty:
                util_row = grp
            row["util_csv"] = util_row["util_csv"].dropna().iloc[0] if not util_row["util_csv"].dropna().empty else ""
        if "ncu_csv" in grp.columns:
            ncu_row = grp[grp["profiler_level"].str.lower() == "level_c"]
            if ncu_row.empty:
                ncu_row = grp
            row["ncu_csv"] = ncu_row["ncu_csv"].dropna().iloc[0] if not ncu_row["ncu_csv"].dropna().empty else ""
        if "kernel_summary_csv" in grp.columns:
            k_row = grp[grp["profiler_level"].str.lower() == "level_c"]
            if k_row.empty:
                k_row = grp
            row["kernel_summary_csv"] = (
                k_row["kernel_summary_csv"].dropna().iloc[0] if not k_row["kernel_summary_csv"].dropna().empty else ""
            )
        collapsed_rows.append(row)
    return pd.DataFrame(collapsed_rows)


def plot_four_metrics_row(
    df: pd.DataFrame,
    out_png: Path,
    *,
    keep_axis_labels: bool = False,
    embed_legend: bool = False,
) -> Dict[str, dict]:
    # Orders
    if "quality" in df.columns:
        df["quality"] = pd.Categorical(df["quality"], categories=QUALITY_ORDER, ordered=True)
    # Stable, deterministic component ordering with explicit priority
    preferred_order = [
        "rfdiffusion",
        "proteinmpnn",
        "esm",
        "mmseqs",
        "protenix",
        "esmfold",
        "alphafold3",
        "diffdock",  # place diffdock before vina variants
        "vina_gpu",
        "vina_cpu",
    ]
    allowed_canon = {_canon_component_name(name) for name in preferred_order}

    def _component_allowed(raw_name: str) -> bool:
        canon = _canon_component_name(raw_name)
        if canon not in allowed_canon:
            return False
        if canon == "alphafold":
            normalized = re.sub(r"[^a-z0-9]", "", str(raw_name).lower())
            return "alphafold3" in normalized
        return True

    # Filter out any unexpected components (e.g., obs2) while allowing AlphaFold3 explicitly
    df = df[df["component"].map(_component_allowed)].copy()
    if df.empty:
        raise SystemExit("No supported components available after filtering unexpected entries.")
    present = list(dict.fromkeys(df["component"].astype(str).tolist()))
    def _canon(name: str) -> str:
        # remove non-letters for robust matching (e.g., mmseqs2 -> mmseqs)
        return _canon_component_name(name)
    canon_to_name = {}
    for n in present:
        k = _canon(n)
        if k not in canon_to_name:
            canon_to_name[k] = n
    desired = []
    for p in preferred_order:
        k = _canon(p)
        if k in canon_to_name:
            desired.append(canon_to_name[k])
    remaining = [n for n in present if n not in desired]
    comp_order = desired + remaining
    df["component"] = pd.Categorical(df["component"], categories=comp_order, ordered=True)

    # Aggregate over sample_id and keep component, quality
    agg_cols = [
        "wall_clock_s",
        "temporal_util_percent",
        "spatial_util_percent",
        "peak_memory_mib",
    ]
    # Keep all qualities and aggregate with quality dimension for hatching
    gdf = (
        df.groupby(["sample_id", "component", "quality"], as_index=False, observed=False)[agg_cols]
        .mean(numeric_only=True)
        .sort_values(["sample_id", "component", "quality"])
    )

    # Convert peak memory to GiB for plotting
    if "peak_memory_mib" in gdf.columns:
        try:
            gdf["peak_memory_gib"] = gdf["peak_memory_mib"] / 1024.0
        except Exception:
            gdf["peak_memory_gib"] = np.nan

    metrics = [
        ("wall_clock_s", "Wall Clock (s)"),
        ("temporal_util_percent", "Temporal Utilization (%)"),
        ("spatial_util_percent", "Spatial Utilization (%)"),
        ("peak_memory_gib", "Peak VRAM (GiB)"),
    ]

    # Figure sizing: 4 rows, width grows with number of samples and components
    n_comp = len(comp_order) if comp_order else 6
    sample_order = list(dict.fromkeys(gdf["sample_id"].astype(str).tolist()))
    if sample_order and LIGAND_ORDER:
        ligand_rank = {lig: idx for idx, lig in enumerate(LIGAND_ORDER)}

        def _split_sample_id(sample: str) -> Tuple[str, str]:
            text = str(sample)
            lowered = text.lower()
            for lig in LIGAND_ORDER:
                if lowered == lig:
                    return "", lig
                for sep in ("_", "-"):
                    suffix = f"{sep}{lig}"
                    if lowered.endswith(suffix):
                        return text[: -len(suffix)], lig
            if "_" in text:
                head, tail = text.rsplit("_", 1)
                return head, tail
            if "-" in text:
                head, tail = text.rsplit("-", 1)
                return head, tail
            return "", text

        scaffold_rank: Dict[str, int] = {}
        for sample in sample_order:
            scaffold, _ = _split_sample_id(sample)
            if scaffold not in scaffold_rank:
                scaffold_rank[scaffold] = len(scaffold_rank)
        existing_index = {name: idx for idx, name in enumerate(sample_order)}

        def _sample_sort_key(sample: str) -> Tuple[int, int, int]:
            scaffold, ligand = _split_sample_id(sample)
            lig_rank = ligand_rank.get(ligand.lower(), len(ligand_rank))
            return (scaffold_rank.get(scaffold, len(scaffold_rank)), lig_rank, existing_index[sample])

        sample_order = sorted(sample_order, key=_sample_sort_key)
    n_groups = max(1, len(sample_order))
    width = max(16, int(2.2 * n_groups + 0.9 * n_comp))
    height = 8.6  # trim overall height so the figure is a bit flatter

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(4, 1, figsize=(width, height), sharex=True, gridspec_kw={"hspace": 0.16})

    # Color per component
    comp_color_map: Dict[str, str] = load_component_color_map(comp_order)
    # Fallback palette for any missing components
    remaining = [c for c in comp_order if c not in comp_color_map]
    if remaining:
        comp_palette = sns.color_palette("tab20", n_colors=max(len(remaining), 3))
        for i, c in enumerate(remaining):
            comp_color_map[c] = comp_palette[i % len(comp_palette)]
    # Hatch per quality (denser for better visibility)
    hatch_map = {"low": "//", "medium": None, "high": "xx"}

    # Bar geometry: use a fixed slot sequence per group to keep x positions
    # consistent across all subplots/metrics.
    cluster_w = 0.9
    pad_fraction = 0.05  # slightly tighter outer margins

    # Build x positions for sample groups
    group_keys = sample_order

    # Spacing factor between groups (used for bars and ticks)
    group_gap = cluster_w * 0.18  # extra breathing room between samples
    group_spacing = cluster_w + group_gap

    # Build a fixed, hard-coded set of (component, quality) slots per group.
    # Components like proteinmpnn and Vina-GPU only plot the medium setting; MMSeqs adds high.
    def _canon(name: str) -> str:
        return _canon_component_name(name)
    fixed_quals_by_comp: Dict[str, List[str]] = {}
    for comp in comp_order:
        ck = _canon(comp)
        if ck in ("proteinmpnn", "vinagpu"):
            fixed_quals_by_comp[comp] = ["medium"]
        elif ck == "mmseqs":
            fixed_quals_by_comp[comp] = ["medium", "high"]
        else:
            fixed_quals_by_comp[comp] = QUALITY_ORDER[:]
    slot_defs: List[Tuple[str, str]] = []  # list of (component, quality)
    for comp in comp_order:
        for q in fixed_quals_by_comp.get(comp, QUALITY_ORDER):
            slot_defs.append((comp, q))
    n_slots = max(1, len(slot_defs))
    slot_barw = cluster_w / n_slots

    for ax, (metric, _title) in zip(axes, metrics):
        sub = gdf.copy()
        if sub[metric].notna().sum() == 0:
            ax.set_visible(False)
            continue
        group_bounds: Dict[int, Tuple[float, float]] = {}
        for gi, sample in enumerate(group_keys):
            cell = sub[(sub["sample_id"] == sample)]
            x0 = gi * group_spacing
            if cell.empty:
                continue
            start_x = x0 - cluster_w / 2
            bar_entries: List[Tuple[float, float, str, str, bool]] = []
            for j, (comp, q) in enumerate(slot_defs):
                rows = cell[(cell["component"] == comp) & (cell["quality"] == q)]
                bx = start_x + j * slot_barw + slot_barw / 2
                present = False
                val = 0.0
                if not rows.empty:
                    raw_val = rows.iloc[0][metric]
                    if pd.notna(raw_val):
                        try:
                            val = float(raw_val)
                        except Exception:
                            val = 0.0
                        present = True
                bar_entries.append((bx, val, comp, q, present))
            if not any(entry[4] for entry in bar_entries):
                continue
            group_bounds[gi] = (start_x, start_x + cluster_w)
            for bx, val, comp, q, present in bar_entries:
                if not present:
                    continue
                ax.bar(
                    bx,
                    val,
                    width=slot_barw,
                    color=comp_color_map[comp],
                    edgecolor="black",
                    linewidth=0.8,
                    hatch=hatch_map.get(q, None),
                )
                left_edge = bx - slot_barw / 2
                right_edge = bx + slot_barw / 2
                if gi in group_bounds:
                    prev_left, prev_right = group_bounds[gi]
                    group_bounds[gi] = (min(prev_left, left_edge), max(prev_right, right_edge))
                else:
                    group_bounds[gi] = (left_edge, right_edge)
        # Draw vertical separators between active scaffold/ligand groups for readability
        sorted_groups = sorted(group_bounds)
        if len(sorted_groups) > 1:
            for left_idx, right_idx in zip(sorted_groups, sorted_groups[1:]):
                left_bounds = group_bounds[left_idx]
                right_bounds = group_bounds[right_idx]
                boundary_center = (left_bounds[1] + right_bounds[0]) / 2.0
                ax.axvline(
                    boundary_center,
                    color="#bdbdbd",
                    linestyle="-",
                    linewidth=0.8,
                    zorder=0,
                )
        # Tighten left/right padding using actual bar bounds so edges feel even
        pad = pad_fraction * group_spacing
        if group_bounds:
            min_left = min(bounds[0] for bounds in group_bounds.values())
            max_right = max(bounds[1] for bounds in group_bounds.values())
        else:
            min_left = -cluster_w / 2
            max_right = (n_groups - 1) * group_spacing + cluster_w / 2
        ax.set_xlim(min_left - pad, max_right + pad)
        ax.margins(x=0)
        if keep_axis_labels:
            ax.set_ylabel(_title)
        # Paper-like axis styling
        ax.grid(axis="y", color="#b0b0b0", linewidth=0.6, alpha=0.6)
        ax.grid(axis="x", visible=False)
        # Black border on all sides
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1.2)
            try:
                ax.spines[side].set_color("black")
            except Exception:
                pass
        ax.tick_params(axis="both", direction="out", length=4, width=1.0, pad=2)
        # Enlarge y tick label size relative to rcParams baseline
        try:
            ytick_fs = float(plt.rcParams.get("ytick.labelsize", 14)) + 3
        except Exception:
            ytick_fs = 17
        ax.tick_params(axis="y", labelsize=ytick_fs)
        # Utilization axes fixed to 0-100
        if metric in ("temporal_util_percent", "spatial_util_percent"):
            ax.set_ylim(0, 100)
            try:
                ax.set_yticks([0, 20, 40, 60, 80, 100])
            except Exception:
                pass
        # Cap wall-clock at 200s for readability
        if metric == "wall_clock_s" and not keep_axis_labels:
            try:
                ax.set_ylim(0, 200)
                ax.set_yticks([0, 50, 100, 150, 200])
            except Exception:
                pass

    if keep_axis_labels:
        centers = [gi * group_spacing for gi in range(n_groups)]
        labels = [str(s) for s in group_keys]
        axes[-1].set_xticks(centers)
        axes[-1].set_xticklabels(labels, rotation=45, ha="right")
        axes[-1].set_xlabel("Sample")
    else:
        # Remove shared x-axis ticks and labels per request
        axes[-1].set_xticks([])
        axes[-1].set_xticklabels([])
        axes[-1].set_xlabel("")
        for ax in axes:
            ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    legend_info = {
        "comp_order": {"order": comp_order, "color_map": comp_color_map},
        "quality": {"order": QUALITY_ORDER, "hatch_map": hatch_map},
        "figure": {"width": width},
    }

    if embed_legend:
        from matplotlib.patches import Patch

        def pretty_label(name: str) -> str:
            n = str(name)
            key = re.sub(r"[^a-z]", "", n.lower())
            mapping = {
                "rfdiffusion": "RFDiffusion",
                "proteinmpnn": "ProteinMPNN",
                "esm": "ESM-2",
                "mmseqs": "MMSeqs2",
                "protenix": "Protenix",
                "alphafold": "AlphaFold3",
                "esmfold": "ESMFold",
                "vinacpu": "Vina-CPU",
                "vinagpu": "Vina-GPU",
                "diffdock": "DiffDock",
            }
            return mapping.get(key, n)

        quality_order = legend_info["quality"]["order"]
        hatch_map = legend_info["quality"]["hatch_map"]
        comp_handles = [
            Patch(facecolor=comp_color_map[c], edgecolor="black", label=pretty_label(c))
            for c in comp_order
        ]
        qual_label_map = {"low": "Throughput", "medium": "Balanced", "high": "Accuracy"}
        quality_handles = [
            Patch(facecolor="white", edgecolor="black", hatch=hatch_map[q], label=qual_label_map.get(q, q))
            for q in quality_order
        ]
        handles = comp_handles + quality_handles
        ncols = max(3, min(len(handles), 6))
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=ncols,
            frameon=True,
            bbox_to_anchor=(0.5, 1.05),
            handlelength=2.0,
            handleheight=0.6,
            handletextpad=1.0,
            columnspacing=1.6,
        )

    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    # Return info for external legend construction
    return legend_info


def save_combined_legend(legend_info: Dict[str, dict], out_png: Path):
    from matplotlib.patches import Patch
    # Pretty display names for components in legend
    def pretty_label(name: str) -> str:
        n = str(name)
        key = re.sub(r"[^a-z]", "", n.lower())
        mapping = {
            "rfdiffusion": "RFDiffusion",
            "proteinmpnn": "ProteinMPNN",
            "esm": "ESM-2",
            "mmseqs": "MMSeqs2",
            "protenix": "Protenix",
            "alphafold": "AlphaFold3",
            "esmfold": "ESMFold",
            "vinacpu": "Vina-CPU",
            "vinagpu": "Vina-GPU",
            "diffdock": "DiffDock",
        }
        return mapping.get(key, n)
    comp_order = legend_info["comp_order"]["order"]
    comp_color_map = legend_info["comp_order"]["color_map"]
    quality_order = legend_info.get("quality", {}).get("order", QUALITY_ORDER)
    hatch_map = legend_info.get("quality", {}).get("hatch_map", {"low": "//", "medium": None, "high": "xx"})

    # Paper labels for quality
    qual_label_map = {"low": "Throughput", "medium": "Balanced", "high": "Accuracy"}
    quality_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=hatch_map[q], label=qual_label_map.get(q, q))
        for q in quality_order
    ]
    comp_handles = [
        Patch(facecolor=comp_color_map[c], edgecolor="black", label=pretty_label(c))
        for c in comp_order
    ]

    # Components first, then qualities; force single line
    handles = comp_handles + quality_handles
    ncols = len(handles)
    nrows = (len(handles) + ncols - 1) // ncols
    handle_length = 2.0  # shrink bar length inside legend handles
    handle_height = 0.65  # shrink bar height slightly
    handle_text_pad = 1.2
    column_spacing = 3.2
    base_width = max(12, 2.3 * ncols)
    fig_width = legend_info.get("figure", {}).get("width")
    try:
        fig_width_val = float(fig_width) if fig_width is not None else None
    except (TypeError, ValueError):
        fig_width_val = None
    per_handle_width = (handle_length + handle_text_pad + column_spacing + 0.6) * ncols
    width_candidates = [base_width, per_handle_width]
    if fig_width_val:
        width_candidates.append(fig_width_val * 2.0)
    width = max(width_candidates)
    height = 1.05 + 0.35 * max(1, nrows)
    fig = plt.figure(figsize=(width, height))
    leg = fig.legend(
        handles=handles,
        loc="center",
        ncol=ncols,
        frameon=True,
        bbox_to_anchor=(0.5, 0.5),
        handlelength=handle_length,
        handleheight=handle_height,
        handletextpad=handle_text_pad,
        columnspacing=column_spacing,
        borderaxespad=0.6,
    )
    frame = leg.get_frame()
    frame.set_edgecolor("black")
    frame.set_linewidth(max(1.25, frame.get_linewidth() * 1.5))
    try:
        frame.set_boxstyle("square", pad=0.2)
    except Exception:
        # Fallback for older Matplotlib where square boxstyle is unavailable
        frame.set_linewidth(frame.get_linewidth())

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def apply_paper_style():
    # Base font family preferences; allow override via env
    font_path = os.environ.get("PAPER_FONT_PATH")
    if font_path and Path(font_path).exists():
        try:
            font_manager.fontManager.addfont(font_path)
        except Exception:
            pass
    plt.rcParams["font.family"] = [
        "Helvetica Neue",
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
        "sans-serif",
    ]
    base = 14
    mpl_params = {
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.6,
        "hatch.linewidth": 0.35,
    }
    for k, v in mpl_params.items():
        plt.rcParams[k] = v


def main():
    ap = argparse.ArgumentParser(description="Generate profiling graphs (wide 1x4 figure)")
    ap.add_argument(
        "--data-root",
        dest="data_roots",
        action="append",
        help="Root directory containing component profiling outputs. Repeat to combine multiple roots. "
        "If omitted, defaults to a preset list (parsed_csv, legacy snapshot, esm workspace).",
    )
    ap.add_argument(
        "--experiments",
        "--experiment",
        "--expreriments",
        dest="experiments",
        help="Experiments YAML name or path (e.g., minimal.yaml) used to filter component/sample_id/quality.",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="profiling/plots/outputs",
        help="Output directory for figures and CSV",
    )
    ap.add_argument(
        "--debug-inputs",
        action="store_true",
        help="Print every utilization/kernel/ncu file discovered while parsing inputs.",
    )
    ap.add_argument(
        "--debug-groups",
        action="store_true",
        help="Print the unique scaffold/ligand size combinations that will be rendered on the x-axis.",
    )
    args = ap.parse_args()

    if args.data_roots:
        data_roots = [Path(p).expanduser() for p in args.data_roots]
    else:
        data_roots = [Path(p).expanduser() for p in DEFAULT_DATA_ROOTS]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Apply global paper-like style
    apply_paper_style()

    exp_filters = load_experiment_filters(args.experiments)
    df = collect_records(data_roots, verbose=args.debug_inputs, progress=True)
    if exp_filters:
        before = len(df)
        df = filter_records_by_experiments(df, exp_filters)
        if args.debug_inputs:
            print(f"[FILTER] experiments={args.experiments} kept {len(df)}/{before} rows")
    df = collapse_metrics(df)
    if args.debug_groups and not df.empty:
        samples = sorted(df["sample_id"].dropna().unique())
        print("[DEBUG] Unique sample_id groups (x-axis):")
        for idx, sid in enumerate(samples):
            print(f"  {idx + 1:02d}: sample_id={sid}")

    # For Vina CPU, force peak VRAM to 0
    if not df.empty and "component" in df.columns:
        comp_lower = df["component"].astype(str).str.lower()
        mask_vina_cpu = comp_lower.str.contains("vina") & comp_lower.str.contains("cpu")
        if "peak_memory_mib" in df.columns:
            df.loc[mask_vina_cpu, "peak_memory_mib"] = 0.0
    if df.empty:
        if exp_filters:
            print(
                "No profiling records matched experiments",
                args.experiments,
                "under roots:",
                ", ".join(str(p) for p in data_roots),
            )
        else:
            print("No profiling records found under:", ", ".join(str(p) for p in data_roots))
        return

    # Save summary CSV
    summary_csv = outdir / "summary_metrics.csv"
    df.to_csv(summary_csv, index=False)
    print("Wrote", summary_csv)

    # Generate 4x1 figure, then export separate legend figure
    overview_png = outdir / "metrics_overview.png"
    legend_info = plot_four_metrics_row(df, overview_png, keep_axis_labels=False, embed_legend=False)
    overview_debug_png = outdir / "metrics_overview_debug.png"
    plot_four_metrics_row(df, overview_debug_png, keep_axis_labels=True, embed_legend=True)
    legend_png = outdir / "metrics_legend.png"
    save_combined_legend(legend_info, legend_png)
    print("Wrote", overview_png)
    print("Wrote", overview_debug_png)
    print("Wrote", legend_png)


if __name__ == "__main__":
    main()
