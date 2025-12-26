#!/usr/bin/env python3
"""Collect per-run profiling features into a tidy CSV by walking raw profiling outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml

DEFAULT_RAW_ROOT = Path("/mnt/nfs/new/bioinformatics/profile/raw")
DEFAULT_SAMPLES = Path("profiling/configs/samples.yaml")
DEFAULT_COMPONENTS_DIR = Path("profiling/configs/components")
DEFAULT_OUTDIR = Path("modeling")
DEFAULT_LEVEL = "level_a"
DEFAULT_OUTFILE = "features.csv"

_COMPONENT_KNOB_KEYS: Dict[str, set[str]] = {
    "rfdiffusion": {"num_designs", "step", "steps"},
    "proteinmpnn": {"num_seq_per_target"},
    "esm": {"num_variants", "max_mutations", "top_k"},
    "protenix": {"num_samples", "step", "cycle"},
    "diffdock": {"batch_size", "samples_per_complex", "steps", "inference_steps"},
    "vina_gpu": {"batch_size"},
    "vina_cpu": {"batch_size"},
    "alphafold3": set(),
    "esmfold": set(),
    "mmseqs2": set(),
}
_GENERIC_NUMERIC_KEYS = {
    "batch_size",
    "samples_per_complex",
    "num_designs",
    "num_seq_per_target",
    "num_variants",
    "num_samples",
    "steps",
    "step",
    "cycle",
}
_SHARED_KNOB_MAP = {
    "batch_size": "input_batch_size",
    "output_samples": "output_samples",
}
_IGNORED_LABEL_KEYS = {
    "sample_id",
    "quality",
    "scaffold",
    "ligand",
    "dataset",
    "run_label",
    "repeat_idx",
    "repeat",
    "contig",
    "quality_level",
    "total_samples",
    "input_batch_size",
    "output_samples",
    "assigned_gpu",
}


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def load_component_batching_aliases(components_dir: Optional[Path]) -> Dict[str, Dict[str, str]]:
    """Load batching key aliases from profiling component YAMLs.

    Profiling uses experiment-level batch knobs (`input_batch_size`, `output_samples`) which
    are mapped into component-specific keys via each component spec's `batching.{input,output}.key`.
    Reading those specs lets us avoid emitting redundant columns like `diffdock_samples_per_complex`
    when they are simply aliases of `output_samples`.
    """

    if components_dir is None:
        return {}
    if not components_dir.exists():
        _warn(f"Components config directory not found: {components_dir}")
        return {}

    aliases: Dict[str, Dict[str, str]] = {}
    yaml_paths = sorted(list(components_dir.glob("*.yaml")) + list(components_dir.glob("*.yml")))
    for yaml_path in yaml_paths:
        try:
            data = yaml.safe_load(yaml_path.read_text()) or {}
        except Exception as exc:  # pragma: no cover - defensive parsing
            _warn(f"Failed to parse component config {yaml_path}: {exc}")
            continue
        if not isinstance(data, Mapping):
            continue

        component = str(data.get("name") or yaml_path.stem)
        batching = data.get("batching") if isinstance(data.get("batching"), Mapping) else None

        input_key = None
        output_key = None
        if batching:
            input_node = batching.get("input") if isinstance(batching.get("input"), Mapping) else None
            output_node = batching.get("output") if isinstance(batching.get("output"), Mapping) else None
            if input_node and input_node.get("key"):
                input_key = str(input_node.get("key"))
            if output_node and output_node.get("key"):
                output_key = str(output_node.get("key"))

        mapping: Dict[str, str] = {}
        if input_key:
            mapping[input_key] = "input_batch_size"
        if output_key:
            mapping[output_key] = "output_samples"

        if mapping:
            aliases[component] = mapping

    return aliases


def _normalize_level(level: str) -> str:
    raw = (level or "").strip().lower()
    if not raw:
        return DEFAULT_LEVEL
    if not raw.startswith("level_"):
        raw = f"level_{raw}"
    return raw


def _ensure_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _maybe_int(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, float) and not math.isnan(value):
        try:
            return int(value)
        except Exception:
            return value
    if isinstance(value, str):
        raw = value.strip()
        if raw.lstrip("+-").isdigit():
            try:
                return int(raw)
            except Exception:
                return raw
    return value


def _coerce_numeric(value) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value) if float(value).is_integer() else float(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            int_val = int(raw)
            return int_val
        except Exception:
            try:
                float_val = float(raw)
                return int(float_val) if float_val.is_integer() else float_val
            except Exception:
                return None
    try:
        float_val = float(value)
        return int(float_val) if float_val.is_integer() else float_val
    except Exception:
        return None


def load_sample_metadata(path: Path) -> Dict[str, Dict[str, Optional[int]]]:
    if not path.exists():
        _warn(f"Sample metadata not found: {path}")
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception as exc:  # pragma: no cover - defensive parsing
        _warn(f"Failed to parse sample metadata {path}: {exc}")
        return {}
    groups = data.get("samples") if isinstance(data, Mapping) else None
    if not isinstance(groups, Mapping):
        _warn(f"Sample metadata does not include 'samples' mapping: {path}")
        return {}

    meta: Dict[str, Dict[str, Optional[int]]] = {}
    for entries in groups.values():
        if not isinstance(entries, Sequence):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            sample_id = entry.get("sample_id")
            if not sample_id:
                continue
            scaffold_len = entry.get("scaffold_length")
            if scaffold_len is None:
                scaffold_len = entry.get("scaffold_lenght")
            ligand_len = entry.get("ligand_length")
            meta[str(sample_id)] = {
                "scaffold_length": _coerce_numeric(scaffold_len),
                "ligand_length": _coerce_numeric(ligand_len),
            }
    return meta


def infer_lengths_from_sample_id(sample_id: str) -> Tuple[Optional[int], Optional[int]]:
    scaffold_length: Optional[int] = None
    ligand_length: Optional[int] = None
    tokens = str(sample_id).split("_")
    if tokens and tokens[-1].isdigit():
        try:
            ligand_length = int(tokens[-1])
        except Exception:
            ligand_length = None
    return scaffold_length, ligand_length


def run_is_successful(run_dir: Path) -> bool:
    status_file = run_dir / "profile_status.json"
    if status_file.exists() and status_file.is_file():
        try:
            data = json.loads(status_file.read_text())
            status = str(data.get("status", "")).lower()
            if status:
                return status == "success"
        except Exception:
            _warn(f"Failed to parse status file {status_file}; assuming failure.")
            return False
    success_marker = run_dir / "profile_SUCCESS"
    return success_marker.exists() and success_marker.is_file()


def find_latest_telemetry(run_dir: Path) -> Optional[Path]:
    telemetry_files = [p for p in run_dir.glob("*_telemetry.csv") if p.is_file()]
    if not telemetry_files:
        return None
    try:
        return max(telemetry_files, key=lambda p: p.stat().st_mtime)
    except OSError:
        telemetry_files.sort()
        return telemetry_files[-1] if telemetry_files else None


def parse_telemetry_metrics(path: Path) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Parse wall clock, temporal utilization, and peak memory from telemetry CSV."""
    try:
        rows = list(csv.reader(path.open()))
    except Exception as exc:
        _warn(f"Failed to read telemetry {path}: {exc}")
        return None, None, None
    if len(rows) <= 1:
        return None, None, None

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

    wall_clock_s: Optional[float]
    temporal_util: Optional[float]
    peak_mem: Optional[float]
    if timestamps:
        try:
            wall_clock_s = float((max(timestamps) - min(timestamps)).total_seconds())
        except Exception:
            wall_clock_s = None
    else:
        wall_clock_s = None
    temporal_util = float(pd.Series(per_row_util).mean()) if per_row_util else None
    peak_mem = float(pd.Series(per_row_mem).max()) if per_row_mem else None
    return wall_clock_s, temporal_util, peak_mem


def _collect_label_mappings(data: Mapping) -> List[Mapping]:
    label_sources: List[Mapping] = []
    run_section = data.get("run") if isinstance(data.get("run"), Mapping) else None
    for key in ("labels", "combo", "meta", "inputs", "batch"):
        node = None
        if run_section and isinstance(run_section.get(key), Mapping):
            node = run_section[key]
        if node is None and isinstance(data.get(key), Mapping):
            node = data.get(key)
        if node is not None:
            label_sources.append(node)
    return label_sources


def _parse_dir_value(name: str, prefix: str) -> Optional[int]:
    stem = name.strip()
    expected = f"{prefix}_"
    if not stem.startswith(expected):
        return None
    raw = stem[len(expected) :]
    try:
        return int(raw)
    except Exception:
        return None


def _iter_quality_roots(sample_dir: Path) -> List[Tuple[str, Path]]:
    """Yield (quality_label, root_dir) pairs where root_dir contains input_* directories.

    Most components store runs as:
        <sample>/<quality>/input_<n>/output_<m>/run_<k>
    Some legacy proteinmpnn runs omit the quality directory:
        <sample>/input_<n>/output_<m>/run_<k>
    """

    direct_input_dirs = sorted(p for p in sample_dir.glob("input_*") if p.is_dir())

    quality_dirs: List[Path] = []
    for child in sorted(p for p in sample_dir.iterdir() if p.is_dir()):
        if child.name.startswith((".", "input_", "output_")):
            continue
        if any(p.is_dir() for p in child.glob("input_*")):
            quality_dirs.append(child)

    roots: List[Tuple[str, Path]] = []
    if direct_input_dirs:
        quality_names = {p.name for p in quality_dirs}
        if "medium" in quality_names:
            default_quality = "medium"
        elif len(quality_dirs) == 1:
            default_quality = quality_dirs[0].name
        else:
            default_quality = "unlabeled"
        roots.append((default_quality, sample_dir))

    roots.extend((p.name, p) for p in quality_dirs)
    return roots


def extract_component_knobs(
    component: str,
    run_dir: Path,
    *,
    shared_knob_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, float]:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        cfg = yaml.safe_load(config_path.read_text()) or {}
    except Exception as exc:  # pragma: no cover - defensive parsing
        _warn(f"Failed to parse config {config_path}: {exc}")
        return {}
    if not isinstance(cfg, Mapping):
        return {}

    knobs: Dict[str, float] = {}
    allowed_keys = set(_GENERIC_NUMERIC_KEYS)
    allowed_keys.update(_COMPONENT_KNOB_KEYS.get(component, set()))
    label_sources = _collect_label_mappings(cfg)
    shared_map = dict(_SHARED_KNOB_MAP)
    if shared_knob_map:
        shared_map.update({str(k): str(v) for k, v in shared_knob_map.items()})

    for src in label_sources:
        for key, value in src.items():
            if key in _IGNORED_LABEL_KEYS:
                continue
            col_name = shared_map.get(key)
            if col_name is None:
                if key not in allowed_keys and not key.startswith("num_"):
                    continue
                col_name = f"{component}_{key}"
            numeric_val = _coerce_numeric(value)
            if numeric_val is None:
                continue
            if col_name not in knobs:
                knobs[col_name] = numeric_val
    return knobs


def collect_feature_rows(
    raw_root: Path,
    level: str,
    sample_meta: Mapping[str, Mapping[str, Optional[int]]],
    components_dir: Optional[Path] = DEFAULT_COMPONENTS_DIR,
) -> List[Dict]:
    rows: List[Dict] = []
    missing_samples: set[str] = set()
    if not raw_root.exists():
        _warn(f"Raw root does not exist: {raw_root}")
        return rows

    batching_aliases = load_component_batching_aliases(components_dir)

    components = sorted(p for p in raw_root.iterdir() if p.is_dir())
    if not components:
        _warn(f"No component directories under {raw_root}")
        return rows

    for comp_dir in components:
        component = comp_dir.name
        level_dir = comp_dir / level
        if not level_dir.exists():
            _warn(f"Missing level directory for {component}: {level_dir}")
            continue
        samples = sorted(p for p in level_dir.iterdir() if p.is_dir())
        if not samples:
            _warn(f"No samples under {level_dir}")
            continue
        for sample_dir in samples:
            sample = sample_dir.name
            quality_roots = _iter_quality_roots(sample_dir)
            if not quality_roots:
                _warn(f"No input/quality directories under {sample_dir}")
                continue
            for quality, quality_root in quality_roots:
                input_dirs = sorted(p for p in quality_root.glob("input_*") if p.is_dir())
                if not input_dirs:
                    _warn(f"No input_* directories under {quality_root}")
                    continue
                for input_dir in input_dirs:
                    batch_size = _parse_dir_value(input_dir.name, "input")
                    if batch_size is None:
                        _warn(f"Unrecognized input dir name (expected input_<n>): {input_dir}")
                        continue
                    output_dirs = sorted(p for p in input_dir.glob("output_*") if p.is_dir())
                    if not output_dirs:
                        _warn(f"No output_* directories under {input_dir}")
                        continue
                    for output_dir in output_dirs:
                        output_samples = _parse_dir_value(output_dir.name, "output")
                        if output_samples is None:
                            _warn(f"Unrecognized output dir name (expected output_<n>): {output_dir}")
                            continue
                        run_dirs = sorted(p for p in output_dir.glob("run_*") if p.is_dir())
                        if not run_dirs:
                            _warn(f"No run_* directories under {output_dir}")
                            continue
                        for run_dir in run_dirs:
                            if not run_is_successful(run_dir):
                                _warn(f"Skipping failed run (no success marker): {run_dir}")
                                continue
                            telemetry_csv = find_latest_telemetry(run_dir)
                            if telemetry_csv is None:
                                _warn(f"No telemetry CSVs under {run_dir}")
                                continue
                            runtime_sec, temporal_util, peak_mem = parse_telemetry_metrics(telemetry_csv)
                            if runtime_sec is None or math.isnan(runtime_sec):
                                _warn(f"Unable to compute runtime from {telemetry_csv}")
                                continue
                            row: Dict[str, object] = {
                                "component": str(component),
                                "sample_id": str(sample),
                                "quality": str(quality),
                                "input_batch_size": _maybe_int(batch_size),
                                "output_samples": _maybe_int(output_samples),
                                "runtime_sec": float(runtime_sec),
                                "peak_memory_mib": float(peak_mem) if peak_mem is not None else math.nan,
                                "mean_temporal_util_percent": float(temporal_util) if temporal_util is not None else math.nan,
                                "run_path": str(run_dir),
                                "telemetry_csv": str(telemetry_csv),
                                "level": level,
                            }
                            meta = sample_meta.get(str(sample))
                            if meta:
                                row["scaffold_length"] = meta.get("scaffold_length")
                                row["ligand_length"] = meta.get("ligand_length")
                            else:
                                scaffold_len, ligand_len = infer_lengths_from_sample_id(str(sample))
                                row["scaffold_length"] = scaffold_len
                                row["ligand_length"] = ligand_len
                                if str(sample) not in missing_samples:
                                    _warn(f"No sample metadata for {sample}; lengths inferred from sample_id.")
                                    missing_samples.add(str(sample))

                            knob_values = extract_component_knobs(
                                str(component),
                                run_dir,
                                shared_knob_map=batching_aliases.get(str(component)),
                            )
                            row.update(knob_values)
                            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help=f"Root directory containing raw telemetry (default: {DEFAULT_RAW_ROOT})",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=DEFAULT_SAMPLES,
        help=f"Path to samples.yaml with scaffold/ligand lengths (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help=f"Directory to write outputs (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=DEFAULT_OUTFILE,
        help=f"CSV file name to write under outdir (default: {DEFAULT_OUTFILE})",
    )
    parser.add_argument(
        "--level",
        type=str,
        default=DEFAULT_LEVEL,
        help="Profiler level directory name (default: level_a)",
    )
    parser.add_argument(
        "--components-dir",
        type=Path,
        default=DEFAULT_COMPONENTS_DIR,
        help=(
            "Path to profiling component YAMLs (used to alias batching keys like samples_per_complex "
            "to shared columns such as output_samples)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    level = _normalize_level(args.level)
    sample_meta = load_sample_metadata(args.samples)

    rows = collect_feature_rows(args.raw_root, level, sample_meta, components_dir=args.components_dir)
    if not rows:
        _warn("No runs collected; CSV will not be written.")
        return

    df = pd.DataFrame(rows)
    base_cols = [
        "component",
        "sample_id",
        "quality",
        "level",
        "input_batch_size",
        "output_samples",
        "runtime_sec",
        "peak_memory_mib",
        "mean_temporal_util_percent",
        "scaffold_length",
        "ligand_length",
        "run_path",
        "telemetry_csv",
    ]
    knob_cols = sorted([c for c in df.columns if c not in base_cols])
    ordered_cols = base_cols + knob_cols
    df = df.reindex(columns=ordered_cols)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / args.outfile
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    components = ", ".join(sorted(df["component"].unique()))
    print(f"Components covered: {components}")


if __name__ == "__main__":
    main()
