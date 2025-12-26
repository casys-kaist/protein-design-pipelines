"""Shared utilities for profile component sweepers."""

from __future__ import annotations

import ast
import importlib.util
import itertools
import os
import json
import re
import shlex
import signal
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from profile import RAW_DATA_ROOT
from profile.components.batching import BatchDimension, BatchingConfig
from profile.profilers.base import ProfilerStrategy
from profile.utils.samples import lookup_sample
from profile.utils.fingerprint import persist_fingerprint
from profile.utils.denovo_locator import resolve_denovo_artifacts

RunDict = Dict[str, Any]
ApplyRulesFn = Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]
DEFAULT_QUALITY = "medium"


class MissingNsightTraceError(RuntimeError):
    """Raised when Level C is invoked without a prerequisite Nsight Systems trace."""


def load_module_from_path(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def deep_format(obj: Any, variables: Dict[str, Any]) -> Any:
    if isinstance(obj, str):
        return obj.format(**variables)
    if isinstance(obj, list):
        return [deep_format(x, variables) for x in obj]
    if isinstance(obj, tuple):
        return tuple(deep_format(list(obj), variables))
    if isinstance(obj, dict):
        return {k: deep_format(v, variables) for k, v in obj.items()}
    return obj


def iter_combos(sweep_vars: Dict[str, List[Any]]):
    keys = list(sweep_vars.keys())
    products = itertools.product(*(sweep_vars[key] for key in keys))
    for values in products:
        yield dict(zip(keys, values))


def normalize_combos(combos: Optional[Any]) -> List[Dict[str, Any]]:
    """Normalize combos into a list of dicts."""
    if combos is None:
        return []
    if isinstance(combos, Mapping):
        combos_iter: Iterable[Any] = iter_combos(dict(combos))
    elif isinstance(combos, Iterable) and not isinstance(combos, (str, bytes)):
        combos_iter = combos
    else:
        return []

    normalized: List[Dict[str, Any]] = []
    for combo in combos_iter:
        if combo is None:
            continue
        if not isinstance(combo, dict):
            raise TypeError(f"Combos must be mappings, got {type(combo)}")
        normalized.append(dict(combo))
    return normalized


def _coerce_int(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        try:
            return int(stripped)
        except ValueError:
            return value
    return value


def _normalize_env_vars(env_cfg: Any) -> Dict[str, str]:
    if not env_cfg:
        return {}
    if isinstance(env_cfg, dict):
        return {str(k): str(v) for k, v in env_cfg.items()}
    if isinstance(env_cfg, (list, tuple)):
        normalized: Dict[str, str] = {}
        for entry in env_cfg:
            if isinstance(entry, str) and "=" in entry:
                key, value = entry.split("=", 1)
                normalized[str(key)] = str(value)
        return normalized
    if isinstance(env_cfg, str):
        text = env_cfg.strip()
        if not text:
            return {}
        parsed: Any = None
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(text)
                break
            except Exception:
                continue
        if isinstance(parsed, (dict, list, tuple)):
            return _normalize_env_vars(parsed)
        normalized: Dict[str, str] = {}
        for token in text.split(","):
            token = token.strip()
            if "=" in token:
                key, value = token.split("=", 1)
                normalized[str(key)] = str(value)
        return normalized
    return {}


def _normalize_quality_label(value: Any, default: str = DEFAULT_QUALITY) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or default
    return str(value)


def _ensure_quality(meta: Dict[str, Any], default: str = DEFAULT_QUALITY) -> None:
    meta["quality"] = _normalize_quality_label(meta.get("quality"), default=default)


def _coerce_batch_dimension(dim: Any) -> Optional[BatchDimension]:
    if dim is None:
        return None
    if isinstance(dim, BatchDimension):
        return dim
    if isinstance(dim, str):
        return BatchDimension(key=dim)
    if isinstance(dim, dict):
        data = {k: v for k, v in dim.items() if k in {"key", "default", "description"}}
        return BatchDimension(**data)
    return None


def _coerce_batching_config(batching: Any) -> Optional[BatchingConfig]:
    if batching is None:
        return None
    if isinstance(batching, BatchingConfig):
        return batching
    if isinstance(batching, dict):
        return BatchingConfig(
            input=_coerce_batch_dimension(batching.get("input")),
            output=_coerce_batch_dimension(batching.get("output")),
        )
    return None


def _extract_batch_value(dim: Optional[BatchDimension], variables: Dict[str, Any]) -> Optional[Any]:
    if dim is None:
        return None
    value = None
    key = dim.key
    if key:
        if key in variables:
            value = variables[key]
        elif dim.default is not None:
            value = dim.default
            variables[key] = value
    else:
        value = dim.default
    return _coerce_int(value)


def _expected_total(input_size: Optional[Any], output_samples: Optional[Any]) -> Optional[int]:
    factors: List[int] = []
    if isinstance(input_size, int):
        factors.append(input_size)
    if isinstance(output_samples, int):
        factors.append(output_samples)
    if not factors:
        return None
    total = 1
    for value in factors:
        total *= max(value, 0)
    return total


def _resolve_batch_metadata(variables: Dict[str, Any], batching: Any) -> Dict[str, Any]:
    cfg = _coerce_batching_config(batching)
    has_experiment_input = "input_batch_size" in variables
    has_experiment_output = "output_samples" in variables

    if cfg is None:
        if has_experiment_input or has_experiment_output:
            raise ValueError(
                "Experiment provided batch knobs (input_batch_size/output_samples) "
                "but the component does not declare a batching config."
            )
        return {}

    # Route experiment-level batch knobs (input_batch_size/output_samples) into the
    # component-specific keys (e.g., batch_size, samples_per_complex) when those keys are unset.
    if cfg.input and cfg.input.key and cfg.input.key not in variables:
        if "input_batch_size" in variables:
            variables[cfg.input.key] = variables["input_batch_size"]
    if cfg.output and cfg.output.key and cfg.output.key not in variables:
        if "output_samples" in variables:
            variables[cfg.output.key] = variables["output_samples"]

    meta: Dict[str, Any] = {}
    input_size = _extract_batch_value(cfg.input, variables)
    output_samples = _extract_batch_value(cfg.output, variables)

    if has_experiment_input and cfg.input is None:
        raise ValueError("input_batch_size provided but component batching.input is not defined.")
    if has_experiment_input and input_size is None:
        raise ValueError(
            f"input_batch_size '{variables.get('input_batch_size')}' could not be resolved "
            f"for batching input key '{cfg.input.key if cfg.input else None}'."
        )

    # Components without an output batch dimension still use a single logical output (fan-out=1).
    # Keep raw directory layouts consistent by emitting output_samples=1, regardless of whether an
    # experiment provided output_samples (those values are ignored when no output dimension exists).
    if cfg.output is None:
        if input_size is not None:
            output_samples = 1
        else:
            output_samples = None
    elif has_experiment_output and output_samples is None:
        raise ValueError(
            f"output_samples '{variables.get('output_samples')}' could not be resolved "
            f"for batching output key '{cfg.output.key if cfg.output else None}'."
        )

    if input_size is not None:
        meta["input_batch_size"] = input_size
    if output_samples is not None:
        meta["output_samples"] = output_samples

    total = _expected_total(input_size, output_samples)
    if total is not None:
        meta["total_samples"] = total

    return meta


def _batch_dir_segments(batch_meta: Dict[str, Any]) -> List[str]:
    segments: List[str] = []
    if batch_meta.get("input_batch_size") is not None:
        segments.append(f"input_{batch_meta['input_batch_size']}")
    if batch_meta.get("output_samples") is not None:
        segments.append(f"output_{batch_meta['output_samples']}")
    return segments


def _populate_from_manifest(variables: Dict[str, Any]) -> Dict[str, Any]:
    """Fill scaffold/ligand/dataset/contig from samples.yaml when only sample_id is provided."""
    sample_id = variables.get("sample_id")
    if not sample_id:
        return variables
    meta = lookup_sample(sample_id) or {}
    if meta.get("scaffold") and not variables.get("scaffold"):
        variables["scaffold"] = meta["scaffold"]
    if meta.get("ligand") and not variables.get("ligand"):
        variables["ligand"] = meta["ligand"]
    if meta.get("dataset"):
        variables.setdefault("dataset", meta["dataset"])
    if meta.get("contig"):
        variables.setdefault("contig", meta["contig"])
    return variables


def _profiler_bucket(label: Optional[str]) -> str:
    """Normalize profiler label to a stable directory name (e.g., level_a, level_b)."""
    if label is None:
        return "unspecified"
    text = str(label).strip().lower()
    if not text:
        return "unspecified"
    slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if slug.startswith("level_"):
        parts = slug.split("_")
        if len(parts) >= 2:
            return "_".join(parts[:2])
    return slug or "unspecified"


def _run_prep_commands(commands: Iterable[Any]) -> None:
    for cmd in commands or []:
        text = str(cmd).strip()
        if not text:
            continue
        # Prefer the running interpreter for python-based prep steps so container
        # environments that invoke Python via an absolute path don't depend on a
        # `python`/`python3` shim in $PATH.
        python_exec = shlex.quote(sys.executable) if sys.executable else "python3"
        text = re.sub(r"^python3?(?=\s|$)", lambda _: python_exec, text)
        print(f"[prep] {text}")
        subprocess.run(text, shell=True, check=True)


def _component_root(path: Path) -> Path:
    """Best-effort component root under RAW_DATA_ROOT."""
    base = RAW_DATA_ROOT
    try:
        rel = path.relative_to(base)
    except ValueError:
        return path.parent
    parts = rel.parts
    if parts:
        return Path(base) / parts[0]
    return Path(base)


def _relative_within(path: Path, root: Path) -> Path:
    """Return path relative to root; fallback to leaf name if unrelated."""
    try:
        rel = path.relative_to(root)
        return rel if str(rel) != "." else Path(path.name)
    except ValueError:
        return Path(path.name)


def build_runs(
    templates: Dict[str, Any],
    combos: Optional[Any],
    apply_rules: Optional[ApplyRulesFn],
    *,
    timestamp: Optional[str] = None,
    repeats: int = 1,
    batching: Optional[Any] = None,
    profiler_label: Optional[str] = None,
) -> List[RunDict]:
    combo_list = normalize_combos(combos)
    if not combo_list:
        return []

    runs: List[RunDict] = []
    timestamp = str(timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"))
    profiler_dir = _profiler_bucket(profiler_label)
    repeats = max(1, repeats)

    for combo in combo_list:
        combo_with_defaults = dict(combo)
        _ensure_quality(combo_with_defaults)

        processed = apply_rules(combo_with_defaults) if apply_rules else combo_with_defaults
        if processed is None:
            continue

        variables = _populate_from_manifest(dict(processed))
        _ensure_quality(variables)
        batch_meta = _resolve_batch_metadata(variables, batching)
        try:
            base_resolved = deep_format(templates, variables)
        except KeyError as exc:
            missing = exc.args[0] if exc.args else "unknown"
            raise ValueError(
                f"Missing required field '{missing}' in run variables; "
                f"ensure samples.yaml or experiment combos provide scaffold/ligand. "
                f"combo={variables}"
            ) from exc

        base_prof_data_dir = Path(base_resolved["PROF_DATA_DIR"])
        base_basename = base_resolved["BASENAME"]

        data_component_root = _component_root(base_prof_data_dir)
        data_rel_path = _relative_within(base_prof_data_dir, data_component_root)
        batch_segments = _batch_dir_segments(batch_meta)

        for repeat_idx in range(1, repeats + 1):
            repeat_suffix = f"r{repeat_idx}"
            run_label = f"run_{repeat_idx}"
            run_label = str(run_label)

            profiler_root_base = Path(data_component_root or base_prof_data_dir.parent)
            if data_rel_path.parts and data_rel_path.parts[0] == str(profiler_dir):
                profiler_root = profiler_root_base
            else:
                profiler_root = profiler_root_base / str(profiler_dir)
            profiler_scoped_prof_data_dir = profiler_root / data_rel_path
            for segment in batch_segments:
                profiler_scoped_prof_data_dir = profiler_scoped_prof_data_dir / segment
            profiler_scoped_prof_data_dir = profiler_scoped_prof_data_dir / str(run_label)
            timestamped_prof_dir = profiler_scoped_prof_data_dir

            labels = dict(variables)
            labels["run_label"] = run_label
            labels.setdefault("repeat_idx", repeat_idx)
            labels.setdefault("repeat", repeat_idx)

            quality_level = _quality_to_level(labels.get("quality"))
            if quality_level is not None:
                labels["quality_level"] = quality_level

            if batch_meta:
                labels.update(batch_meta)

            scaffold_value = labels.get("scaffold") or processed.get("contig")
            if scaffold_value is not None:
                labels.setdefault("scaffold", scaffold_value)
                labels.setdefault("contig", scaffold_value)

            cfg: RunDict = {
                "CMD": deepcopy(base_resolved["CMD"]),
                "PARAMS": deepcopy(base_resolved["PARAMS"]),
                "PROF_DIR": str(timestamped_prof_dir),
                "PROF_DATA_DIR": str(profiler_scoped_prof_data_dir),
                "BASENAME": f"{base_basename}_{repeat_suffix}",
                "PREP_COMMANDS": deepcopy(base_resolved.get("PREP_COMMANDS", [])),
                "SAMPLE_INTERVAL": base_resolved.get("SAMPLE_INTERVAL"),
                "START_INDEX": base_resolved.get("START_INDEX"),
                "run_idx": len(runs) + 1,
                "repeat_idx": repeat_idx,
                "labels": labels,
                # Backwards-compatible aliases: keep `meta`/`combo` pointing to the same dict.
                "combo": labels,
                "meta": labels,
                "inputs": labels,
                "timestamp": timestamp,
                "run_label": run_label,
            }

            if batch_meta:
                cfg["batch"] = batch_meta

            if cfg["PROF_DIR"] and cfg["BASENAME"]:
                cfg["util_csv"] = os.path.join(cfg["PROF_DIR"], f"{cfg['BASENAME']}_utilization.csv")
            else:
                cfg["util_csv"] = None

            runs.append(cfg)
    return runs


def _quality_to_level(quality: Optional[str]) -> Optional[int]:
    if quality is None:
        return None
    mapping = {"low": 1, "medium": 2, "high": 3}
    return mapping.get(str(quality).lower())


def _profiler_label(profiler_cfg: Optional[Any]) -> Optional[str]:
    """Best-effort name extraction that mirrors run_sweeps._profiler_name."""
    if profiler_cfg is None:
        return None
    if isinstance(profiler_cfg, DictConfig):
        return profiler_cfg.get("name")
    if isinstance(profiler_cfg, dict):
        return profiler_cfg.get("name")
    return getattr(profiler_cfg, "name", None)


def _profiler_target(profiler_cfg: Optional[Any]) -> Optional[str]:
    if profiler_cfg is None:
        return None
    if isinstance(profiler_cfg, DictConfig):
        return profiler_cfg.get("_target_")
    if isinstance(profiler_cfg, dict):
        return profiler_cfg.get("_target_")
    target_cls = getattr(profiler_cfg, "__class__", None)
    if target_cls:
        return f"{target_cls.__module__}.{target_cls.__name__}"
    return None


def _is_level_c_profiler(profiler_cfg: Optional[Any]) -> bool:
    """Identify level_c so we can route to the legacy NCU pipeline."""
    if profiler_cfg is None:
        return False
    label = (_profiler_label(profiler_cfg) or "").lower()
    if label == "level_c_ncu":
        return True
    target = _profiler_target(profiler_cfg) or ""
    return target.endswith("NcuProfilerStrategy")


def _write_run_status(
    prof_dir: str,
    status: str,
    *,
    strategy: Optional[str] = None,
    error: Optional[str] = None,
    raw_dir: Optional[str] = None,
) -> Path:
    """Persist a lightweight status marker so completed/failed runs are discoverable later."""
    def _persist(root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        # Remove stale marker files from previous attempts so reruns don't leave
        # mixed states (e.g., profile_ERROR + profile_SUCCESS).
        for marker in root.glob("profile_*"):
            if marker.name == "profile_status.json" or not marker.is_file():
                continue
            try:
                marker.unlink()
            except OSError:
                pass
        payload = {
            "status": status,
            "strategy": strategy,
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
        }
        status_path = root / "profile_status.json"
        status_path.write_text(json.dumps(payload, indent=2))
        marker = root / f"profile_{status.upper()}"
        marker.write_text(status)
        return status_path

    primary_path = _persist(Path(prof_dir))
    if raw_dir:
        try:
            raw_path = Path(raw_dir)
            if raw_path.resolve() != Path(prof_dir).resolve():
                _persist(raw_path)
        except Exception:
            # Best-effort; don't break success/error recording if raw_dir is unavailable.
            pass
    return primary_path


def patch_globals(module, values: Dict[str, Any]) -> None:
    for key, value in values.items():
        if hasattr(module, key):
            setattr(module, key, value)


def recompute_gpu_profile_paths(
    gpu_profile,
    prof_dir: str,
    prof_data_dir: str,
    basename: str,
    trace_stem: Optional[str] = None,
) -> None:
    base_stem = trace_stem or f"{basename}_trace"
    gpu_profile.NSYS_REP = os.path.join(prof_data_dir, f"{base_stem}.nsys-rep")
    gpu_profile.SQLITE_DB = os.path.join(prof_data_dir, f"{base_stem}.sqlite")
    gpu_profile.NCU_DIR = os.path.join(prof_data_dir, f"{basename}_ncu")
    gpu_profile.KERNEL_CSV = os.path.join(prof_data_dir, f"{basename}_kernel_summary.csv")
    gpu_profile.SUMMARY_CSV = os.path.join(prof_data_dir, f"{basename}_ncu_summary.csv")


def run_system_util(sys_util, cfg: RunDict) -> None:
    ensure_dir(cfg["PROF_DIR"])
    patch_globals(
        sys_util,
        {
            "CMD": cfg["CMD"],
            "PARAMS": cfg["PARAMS"],
            "PROF_DIR": cfg["PROF_DIR"],
            "BASENAME": cfg["BASENAME"],
            "PREP_COMMANDS": cfg["PREP_COMMANDS"],
            "SAMPLE_INTERVAL": cfg["SAMPLE_INTERVAL"],
        },
    )
    print(
        f"\n=== [system_utilization] PROF_DIR={cfg['PROF_DIR']} "
        f"BASENAME={cfg['BASENAME']}"
    )
    print("CMD:", " ".join(cfg["CMD"]))
    print("PARAMS:", " ".join(cfg["PARAMS"]))
    sys_util.run_and_monitor_nvml()


def capture_fingerprint_snapshot(cfg: RunDict, filename: str = "system_meta.json") -> Optional[Path]:
    prof_dir = cfg.get("PROF_DIR")
    if not prof_dir:
        return None
    destination = Path(prof_dir) / filename
    try:
        return persist_fingerprint(destination)
    except Exception as exc:  # pragma: no cover - logging only
        print(f"[WARN] Failed to record system fingerprint: {exc}")
        return None


def run_gpu_profile(
    gpu_profile,
    cfg: RunDict,
    *,
    profiler_cfg: Optional[Any] = None,
    runner_context: Optional[Dict[str, Any]] = None,
) -> None:
    ensure_dir(cfg["PROF_DIR"])
    ensure_dir(cfg["PROF_DATA_DIR"])

    if profiler_cfg:
        profiler_name = _profiler_label(profiler_cfg)
        if _is_level_c_profiler(profiler_cfg):
            if gpu_profile is None:
                raise RuntimeError("Level C profiling requires a gpu_profile module; set `gpu_profile_path` in the component.")
            _run_legacy_level_c(
                gpu_profile,
                cfg,
                profiler_cfg=profiler_cfg,
                runner_context=runner_context or {},
            )
            return
        try:
            _run_strategy_profiler(cfg, profiler_cfg, runner_context or {})
        except subprocess.TimeoutExpired as exc:
            _write_run_status(
                cfg["PROF_DIR"],
                "timeout",
                strategy=profiler_name,
                error=str(exc),
                raw_dir=cfg.get("PROF_DATA_DIR"),
            )
            raise
        except Exception as exc:
            _write_run_status(
                cfg["PROF_DIR"],
                "error",
                strategy=profiler_name,
                error=str(exc),
                raw_dir=cfg.get("PROF_DATA_DIR"),
            )
            raise
        else:
            _write_run_status(
                cfg["PROF_DIR"],
                "success",
                strategy=profiler_name,
                error=None,
                raw_dir=cfg.get("PROF_DATA_DIR"),
            )
        return

    ensure_dir(os.path.join(cfg["PROF_DATA_DIR"], f"{cfg['BASENAME']}_ncu"))
    patch_globals(
        gpu_profile,
        {
            "CMD": cfg["CMD"],
            "PARAMS": cfg["PARAMS"],
            "PROF_DIR": cfg["PROF_DIR"],
            "PROF_DATA_DIR": cfg["PROF_DATA_DIR"],
            "BASENAME": cfg["BASENAME"],
            "START_INDEX": cfg.get("START_INDEX"),
            "PREP_COMMANDS": cfg["PREP_COMMANDS"],
        },
    )
    recompute_gpu_profile_paths(gpu_profile, cfg["PROF_DIR"], cfg["PROF_DATA_DIR"], cfg["BASENAME"])
    ensure_dir(gpu_profile.NCU_DIR)
    print(
        f"\n=== [gpu_profile] PROF_DIR={cfg['PROF_DIR']} "
        f"PROF_DATA_DIR={cfg['PROF_DATA_DIR']} BASENAME={cfg['BASENAME']}"
    )
    print("CMD:", " ".join(cfg["CMD"]))
    print("PARAMS:", " ".join(cfg["PARAMS"]))
    gpu_profile.main()


def _run_legacy_level_c(
    gpu_profile,
    cfg: RunDict,
    *,
    profiler_cfg: Any,
    runner_context: Dict[str, Any],
) -> None:
    profiler_name = _profiler_label(profiler_cfg)
    basename = cfg["BASENAME"]
    prof_dir = cfg["PROF_DIR"]
    prof_data_dir = cfg["PROF_DATA_DIR"]
    run_id = (
        cfg.get("run_id")
        or runner_context.get("runner", {}).get("run_id")
        or basename
    )

    patch_globals(
        gpu_profile,
        {
            "CMD": cfg["CMD"],
            "PARAMS": cfg["PARAMS"],
            "PROF_DIR": prof_dir,
            "PROF_DATA_DIR": prof_data_dir,
            "BASENAME": basename,
            "START_INDEX": cfg.get("START_INDEX"),
            "PREP_COMMANDS": cfg["PREP_COMMANDS"],
        },
    )
    run_cfg = dict(runner_context.get("run_config") or {})
    env_overrides = _normalize_env_vars(run_cfg.get("env"))
    try:
        setattr(gpu_profile, "ENV_OVERRIDES", env_overrides)
    except Exception:
        pass

    trace_stem = str(run_id) if run_id else f"{basename}_trace"
    recompute_gpu_profile_paths(
        gpu_profile,
        prof_dir,
        prof_data_dir,
        basename,
        trace_stem=trace_stem,
    )
    ensure_dir(gpu_profile.NCU_DIR)

    # Explicitly look under level_b (same tree structure) for traces.
    def _as_level_b(path: Path) -> Optional[Path]:
        parts = list(path.parts)
        if "level_c" not in parts:
            return None
        mapped = ["level_b" if part == "level_c" else part for part in parts]
        return Path(*mapped)

    base_trace = Path(gpu_profile.NSYS_REP)
    base_trace_level_b = _as_level_b(base_trace) or base_trace
    trace_candidates = [base_trace_level_b]
    legacy_trace = Path(prof_data_dir) / f"{basename}_trace.nsys-rep"
    legacy_trace_b = _as_level_b(legacy_trace) or legacy_trace
    if legacy_trace_b not in trace_candidates:
        trace_candidates.append(legacy_trace_b)

    # Alternate extensions and locations commonly produced by Nsight Systems.
    def _add_candidates(stem: Path) -> None:
        for ext in (".nsys-rep", ".qdrep", ".qdstrm"):
            candidate = stem.with_suffix(ext)
            if candidate not in trace_candidates:
                trace_candidates.append(candidate)

    prof_dir_trace = Path(prof_dir) / f"{basename}_trace.nsys-rep"
    prof_dir_trace_b = _as_level_b(prof_dir_trace) or prof_dir_trace
    if prof_dir_trace_b not in trace_candidates:
        trace_candidates.append(prof_dir_trace_b)

    for stem in (Path(prof_data_dir) / basename, Path(prof_dir) / basename):
        mapped = _as_level_b(stem) or stem
        _add_candidates(mapped)

    if run_id and str(run_id).strip() != trace_stem:
        alt_trace = Path(prof_data_dir) / f"{run_id}.nsys-rep"
        alt_trace_b = _as_level_b(alt_trace) or alt_trace
        if alt_trace_b not in trace_candidates:
            trace_candidates.append(alt_trace_b)
        alt_prof_trace = Path(prof_dir) / f"{run_id}.nsys-rep"
        alt_prof_trace_b = _as_level_b(alt_prof_trace) or alt_prof_trace
        if alt_prof_trace_b not in trace_candidates:
            trace_candidates.append(alt_prof_trace_b)
        for stem in (Path(prof_data_dir) / str(run_id), Path(prof_dir) / str(run_id)):
            mapped = _as_level_b(stem) or stem
            for ext in (".nsys-rep", ".qdrep", ".qdstrm"):
                candidate = mapped.with_suffix(ext)
                if candidate not in trace_candidates:
                    trace_candidates.append(candidate)

    selected_trace = next((path for path in trace_candidates if path.exists()), None)

    if selected_trace is None:
        # Last resort: glob within the level_b-mapped run directories (recursive) for any Nsight trace.
        search_roots = []
        for root in (Path(prof_data_dir), Path(prof_dir)):
            mapped = _as_level_b(root) or root
            if mapped not in search_roots:
                search_roots.append(mapped)
        for root in search_roots:
            if not root.exists():
                continue
            for ext in ("*.nsys-rep", "*.qdrep", "*.qdstrm"):
                matches = sorted(root.rglob(ext))
                if matches:
                    if run_id:
                        name_matches = [m for m in matches if run_id in m.name]
                        selected_trace = name_matches[0] if name_matches else matches[0]
                    else:
                        selected_trace = matches[0]
                    break
            if selected_trace:
                break

    if selected_trace is None:
        selected_trace = trace_candidates[0]

    if not selected_trace.exists():
        _write_run_status(
            prof_dir,
            "error",
            strategy=profiler_name,
            error="missing nsys-rep",
            raw_dir=prof_data_dir,
        )
        raise MissingNsightTraceError(f"missing Nsight Systems trace: {selected_trace}")

    gpu_profile.NSYS_REP = str(selected_trace)
    gpu_profile.SQLITE_DB = str(Path(str(selected_trace)).with_suffix(".sqlite"))
    if hasattr(gpu_profile, "REQUIRE_EXISTING_NSYS_REP"):
        gpu_profile.REQUIRE_EXISTING_NSYS_REP = True

    run_cfg = dict(runner_context.get("run_config") or {})
    env_overrides = _normalize_env_vars(run_cfg.get("env"))

    profile_config_cls = getattr(gpu_profile, "ProfileConfig", None)
    profile_run_fn = getattr(gpu_profile, "run_profile", None)

    def _missing_artifacts(paths: List[Path]) -> List[Path]:
        missing: List[Path] = []
        seen = set()
        for path in paths:
            p = Path(path)
            resolved = p.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                if not p.exists() or not p.is_file() or p.stat().st_size == 0:
                    missing.append(p)
            except OSError:
                missing.append(p)
        return missing

    if profile_config_cls and profile_run_fn:
        raw_sections: Any = None
        if isinstance(profiler_cfg, DictConfig):
            raw_sections = profiler_cfg.get("target_sections")
            if isinstance(raw_sections, DictConfig):
                raw_sections = OmegaConf.to_container(raw_sections, resolve=True)
        elif isinstance(profiler_cfg, dict):
            raw_sections = profiler_cfg.get("target_sections")
        else:
            raw_sections = getattr(profiler_cfg, "target_sections", None)

        ncu_sections: Optional[List[str]] = None
        if raw_sections:
            if isinstance(raw_sections, str):
                ncu_sections = [raw_sections]
            else:
                try:
                    ncu_sections = [str(item).strip() for item in list(raw_sections) if str(item).strip()]
                except TypeError:
                    value = str(raw_sections).strip()
                    ncu_sections = [value] if value else None

        profile_config = profile_config_cls(
            cmd=list(cfg["CMD"]),
            params=list(cfg["PARAMS"]),
            prof_dir=Path(prof_dir),
            prof_data_dir=Path(prof_data_dir),
            basename=basename,
            start_index=cfg.get("START_INDEX") or 1,
            prep_commands=list(cfg["PREP_COMMANDS"]),
            require_existing_nsys_rep=True,
            create_roofline_report=bool(getattr(gpu_profile, "CREATE_ROOFLINE_REPORT", False)),
            env_overrides=env_overrides,
            trace_path=Path(gpu_profile.NSYS_REP),
            ncu_sections=ncu_sections,
        )

        print(
            f"\n=== [gpu_profile:{profiler_name or 'level_c'}] PROF_DIR={prof_dir} "
            f"PROF_DATA_DIR={prof_data_dir} BASENAME={basename}"
        )
        print("CMD:", " ".join(cfg["CMD"]))
        print("PARAMS:", " ".join(cfg["PARAMS"]))
        print(f"NSYS_REP: {profile_config.paths().nsys_rep}")
        print(f"SQLITE_DB: {profile_config.paths().sqlite_db}")

        try:
            profile_run_fn(profile_config)
        except Exception as exc:
            _write_run_status(
                prof_dir,
                "error",
                strategy=profiler_name,
                error=str(exc),
                raw_dir=prof_data_dir,
            )
            raise
        artifacts = resolve_profiler_artifacts(cfg, profiler_cfg, runner_context)
        missing_artifacts = _missing_artifacts(artifacts)
        if missing_artifacts:
            error_msg = "Missing profiler artifacts: " + ", ".join(str(path) for path in missing_artifacts)
            _write_run_status(
                prof_dir,
                "error",
                strategy=profiler_name,
                error=error_msg,
                raw_dir=prof_data_dir,
            )
            raise RuntimeError(error_msg)

        _write_run_status(
            prof_dir,
            "success",
            strategy=profiler_name,
            error=None,
            raw_dir=prof_data_dir,
        )
        return

    # Fallback should be removed in future; if new API is missing, fail fast.
    raise RuntimeError("gpu_profile module is missing ProfileConfig/run_profile for level_c execution.")


def execute_runs(
    runs: List[RunDict],
    component_name: str,
    sys_util_path: str,
    gpu_profile_path: Optional[str] = None,
    run_system: bool = True,
    run_gpu: bool = True,
    profiler_cfg: Optional[Any] = None,
    runner_context: Optional[Dict[str, Any]] = None,
) -> None:
    sys_util = load_module_from_path(sys_util_path, f"sys_util_{component_name}") if run_system else None
    gpu_profile = (
        load_module_from_path(gpu_profile_path, f"gpu_profile_{component_name}")
        if run_gpu and gpu_profile_path
        else None
    )

    total = len(runs)
    for idx, cfg in enumerate(runs, start=1):
        print("\n" + "=" * 96)
        print(f"SWEEP RUN {idx}/{total}  →  PROF_DIR={cfg['PROF_DIR']}  BASENAME={cfg['BASENAME']}")
        print("=" * 96)

        if run_system and sys_util is not None:
            run_system_util(sys_util, cfg)

        if run_gpu and gpu_profile is not None:
            run_gpu_profile(
                gpu_profile,
                cfg,
                profiler_cfg=profiler_cfg,
                runner_context=runner_context,
            )

    print("\nAll sweep runs finished.")


def _run_strategy_profiler(
    cfg: RunDict,
    profiler_cfg: Any,
    runner_context: Dict[str, Any],
) -> None:
    context = _build_profiler_runtime_context(cfg, runner_context)
    resolved_cfg = _prepare_profiler_cfg(profiler_cfg, context)
    strategy: ProfilerStrategy = instantiate(resolved_cfg)
    original_cmd = _original_command(cfg)
    exit_code_path = Path(cfg["PROF_DIR"]) / "payload_exit_code.txt"
    wrapper_path = Path(cfg["PROF_DIR"]) / "payload_wrapper.sh"
    _write_payload_wrapper(wrapper_path, exit_code_path)
    wrapped_cmd = [str(wrapper_path)] + original_cmd
    output_filename = resolved_cfg.get("output_filename") or f"{cfg['BASENAME']}.profile"
    output_path = Path(cfg["PROF_DATA_DIR"]) / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": context.get("runner", {}).get("run_id"),
        "prof_dir": cfg["PROF_DIR"],
        "raw_dir": cfg["PROF_DATA_DIR"],
        "combo": cfg.get("combo", {}),
        "timestamp": cfg.get("timestamp"),
        "run_config": context.get("run_config", {}),
        "output_path": str(output_path),
    }

    print(
        f"\n=== [gpu_profile:{resolved_cfg.get('name','strategy')}] "
        f"PROF_DIR={cfg['PROF_DIR']} PROF_DATA_DIR={cfg['PROF_DATA_DIR']}"
    )
    print("CMD:", " ".join(original_cmd))

    env, mps_controller = _prepare_run_environment(context, cfg)
    log_path = Path(cfg["PROF_DATA_DIR"]) / f"{cfg['BASENAME']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_cfg = dict(context.get("run_config") or {})
    timeout_sec: Optional[float] = None
    raw_timeout = run_cfg.get("timeout_sec")
    if raw_timeout is not None:
        try:
            timeout_sec = float(raw_timeout)
        except (TypeError, ValueError):
            timeout_sec = None
        if timeout_sec is not None and timeout_sec <= 0:
            timeout_sec = None
    if timeout_sec is not None:
        meta["timeout_sec"] = timeout_sec

    def _kill_process_group(proc: subprocess.Popen, *, grace_sec: float = 10.0) -> None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except PermissionError:
            try:
                proc.terminate()
            except Exception:
                return
        try:
            proc.wait(timeout=grace_sec)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except PermissionError:
            try:
                proc.kill()
            except Exception:
                return
        try:
            proc.wait(timeout=grace_sec)
        except subprocess.TimeoutExpired:
            return
    try:
        _run_prep_commands(cfg.get("PREP_COMMANDS"))
        strategy.pre_run_hook(meta)
        command = strategy.build_command(wrapped_cmd, str(output_path))
        with log_path.open("w") as log_handle:
            proc = subprocess.Popen(
                command,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                if timeout_sec is None:
                    proc.wait()
                else:
                    proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                _kill_process_group(proc)
                raise
        fallback_code = proc.returncode if proc.returncode is not None else 1
        exit_code = _read_exit_code(exit_code_path, fallback_code)
        log_failure_hint = _payload_log_failure_hint(log_path)
        # Guard against silent profiler failures that leave no artifact. Some Nsight
        # setups emit .qdstrm or .qdrep instead of the requested .nsys-rep; accept
        # those alternates if present.
        if not _artifact_exists(output_path):
            base_root = output_path.with_suffix("")
            fallback_exts = [".nsys-rep", ".qdrep", ".qdstrm", ""]
            found = next(
                (
                    candidate
                    for ext in fallback_exts
                    for candidate in [base_root.with_suffix(ext)]
                    if _artifact_exists(candidate)
                ),
                None,
            )
            if found:
                meta["output_path_resolved"] = str(found)
            else:
                # If the profiler failed or never wrote output, surface a clear error.
                if exit_code != 0:
                    raise subprocess.CalledProcessError(exit_code, command)
                raise RuntimeError(f"profiler completed but output is missing: {output_path}")
        if exit_code != 0:
            raise subprocess.CalledProcessError(exit_code, command)
        if log_failure_hint:
            raise RuntimeError(f"payload log indicates failure ({log_failure_hint}); see {log_path}")
    finally:
        strategy.post_run_hook(meta)
        if mps_controller:
            mps_controller.stop()


def _prepare_profiler_cfg(profiler_cfg: Any, context: Dict[str, Any]) -> DictConfig:
    if isinstance(profiler_cfg, DictConfig):
        container = OmegaConf.to_container(profiler_cfg, resolve=False)
    else:
        container = profiler_cfg
    profiler_node = OmegaConf.create({"profiler": container})
    context_cfg = OmegaConf.create(context)
    merged = OmegaConf.merge(context_cfg, profiler_node)
    return merged.profiler


def _prepare_run_environment(
    context: Dict[str, Any],
    cfg: RunDict,
) -> Tuple[Dict[str, str], Optional["MPSController"]]:
    env = os.environ.copy()
    # Avoid shadowing stdlib `profile` inside component workloads.
    env.pop("PYTHONPATH", None)
    run_cfg = dict(context.get("run_config") or {})
    env["PROFILE_RUN_CONCURRENCY"] = str(run_cfg.get("concurrency", 1))
    custom_env = _normalize_env_vars(run_cfg.get("env"))
    for key, value in custom_env.items():
        env[str(key)] = str(value)

    mps_controller: Optional[MPSController] = None
    if run_cfg.get("mps_enabled"):
        mps_controller = MPSController(Path(cfg["PROF_DIR"]))
        env.update(mps_controller.start())

    return env, mps_controller


def _original_command(cfg: RunDict) -> List[str]:
    return [str(x) for x in cfg.get("CMD", [])] + [str(x) for x in cfg.get("PARAMS", [])]


def _build_profiler_runtime_context(
    cfg: RunDict,
    runner_context: Dict[str, Any],
) -> Dict[str, Any]:
    source_context = runner_context or {}
    context = {key: deepcopy(value) for key, value in source_context.items()}
    runner_node = dict(context.get("runner") or {})
    runner_node.setdefault("run_id", cfg.get("run_id") or cfg.get("BASENAME"))
    runner_node.setdefault("prof_dir", cfg.get("PROF_DIR"))
    runner_node.setdefault("prof_data_dir", cfg.get("PROF_DATA_DIR"))
    runner_node.setdefault("basename", cfg.get("BASENAME"))
    context["runner"] = runner_node
    context.setdefault("run_config", source_context.get("run_config", {}))
    return context


def _artifact_exists(path: Path) -> bool:
    """Best-effort existence/size check to ensure profiler produced an output."""
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _payload_log_failure_hint(log_path: Path) -> Optional[str]:
    """Return a best-effort hint if the payload log contains fatal failures."""
    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        return None
    lowered = text.lower()
    fatal_markers = [
        "cuda out of memory",
        "cuda error: out of memory",
        "runtimeerror: cuda out of memory",
        "failed to predict batch",
        "oom when allocating",
    ]
    for marker in fatal_markers:
        if marker in lowered:
            return marker
    return None


def _write_payload_wrapper(script_path: Path, exit_code_path: Path) -> None:
    """Create a small wrapper that records the wrapped command's exit code."""
    script_path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set +e",
            '"$@"',
            "status=$?",
            f"echo ${{status}} > {exit_code_path}",
            "exit ${status}",
            "",
        ]
    )
    script_path.write_text(body)
    script_path.chmod(0o755)


def _read_exit_code(exit_code_path: Path, fallback: int) -> int:
    try:
        if exit_code_path.exists():
            text = exit_code_path.read_text().strip()
            return int(text)
    except Exception:
        pass
    return int(fallback)


class MPSController:
    """Best-effort helper to enable/disable CUDA MPS for colocation experiments."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.env: Dict[str, str] = {}
        self._enabled = False

    def start(self) -> Dict[str, str]:
        if shutil.which("nvidia-cuda-mps-control") is None:
            print("[WARN] CUDA MPS control binary not found; skipping MPS enablement.")
            return {}

        pipe_dir = self.run_dir / "mps" / "pipe"
        log_dir = self.run_dir / "mps" / "log"
        pipe_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        env = {
            "CUDA_MPS_PIPE_DIRECTORY": str(pipe_dir),
            "CUDA_MPS_LOG_DIRECTORY": str(log_dir),
        }
        full_env = os.environ.copy()
        full_env.update(env)

        try:
            subprocess.run(
                ["nvidia-cuda-mps-control", "-d"],
                check=True,
                env=full_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._enabled = True
            self.env = env
            return dict(env)
        except Exception as exc:  # pragma: no cover - best-effort only
            print(f"[WARN] Failed to enable CUDA MPS: {exc}")
            self.env = {}
            self._enabled = False
            return {}

    def stop(self) -> None:
        if not self._enabled:
            return
        full_env = os.environ.copy()
        full_env.update(self.env)
        try:
            subprocess.run(
                ["nvidia-cuda-mps-control"],
                input="quit\n",
                text=True,
                check=True,
                env=full_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:  # pragma: no cover - best-effort only
            print(f"[WARN] Failed to shut down CUDA MPS: {exc}")
        finally:
            self._enabled = False
            self.env = {}


def resolve_profiler_artifacts(
    cfg: RunDict,
    profiler_cfg: Optional[Any],
    runner_context: Optional[Dict[str, Any]] = None,
) -> List[Path]:
    def _dedupe(seq: List[Path]) -> List[Path]:
        seen = set()
        out: List[Path] = []
        for item in seq:
            key = item.resolve()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    prof_data_dir = cfg.get("PROF_DATA_DIR")
    if not prof_data_dir:
        raise RuntimeError("PROF_DATA_DIR is required to resolve profiler artifacts.")
    prof_dir = Path(prof_data_dir)

    if not profiler_cfg or _is_level_c_profiler(profiler_cfg):
        basename = cfg.get("BASENAME") or "run"
        return [
            prof_dir / f"{basename}_kernel_summary.csv",
            prof_dir / f"{basename}_ncu_summary.csv",
        ]

    context = _build_profiler_runtime_context(cfg, runner_context or {})
    resolved_cfg = _prepare_profiler_cfg(profiler_cfg, context)
    artifacts: List[Path] = []
    artifact_nodes = resolved_cfg.get("artifacts") or []
    if isinstance(artifact_nodes, DictConfig):
        artifact_nodes = OmegaConf.to_container(artifact_nodes, resolve=True)  # type: ignore[assignment]
    base_data_dir = prof_dir
    base_prof_dir = prof_dir

    if artifact_nodes:
        for entry in artifact_nodes:
            if isinstance(entry, dict):
                location = entry.get("location", "data")
                relative = entry.get("path") or entry.get("name") or ""
            else:
                location = "data"
                relative = str(entry)
            # Results/raw are unified; keep artifacts under raw even if legacy configs specify "results".
            base = base_data_dir
            artifacts.append(base / relative)
    else:
        filename = resolved_cfg.get("output_filename") or f"{cfg['BASENAME']}.profile"
        artifacts.append(base_data_dir / filename)
        # Backward-compatible fallback for older telemetry naming using BASENAME.
        if filename.endswith("_telemetry.csv") and cfg.get("BASENAME"):
            legacy = base_data_dir / f"{cfg['BASENAME']}_telemetry.csv"
            if legacy != artifacts[0]:
                artifacts.append(legacy)

    return _dedupe(artifacts)
