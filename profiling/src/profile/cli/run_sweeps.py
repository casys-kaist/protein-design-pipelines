#!/usr/bin/env python3
"""Convenience launcher for component profiling sweeps."""

from __future__ import annotations

import importlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import yaml
from collections import defaultdict
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from hydra import main as hydra_main
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

CLI_ROOT = Path(__file__).resolve().parent
PROFILE_ROOT = CLI_ROOT.parent
PROFILING_ROOT = PROFILE_ROOT.parents[1]
REPO_ROOT = PROFILING_ROOT.parent
PACKAGES_ROOT = PROFILING_ROOT / "src"

if str(PACKAGES_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGES_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.modules.pop("profile", None)
from profile.components import ComponentSpec, list_component_names, load_component_spec  # noqa: E402
from profile.components import _shared  # noqa: E402
from profile.cli import notifier as notifier_mod  # noqa: E402
from profile.config import PATHS  # noqa: E402
from profile.layout import OUTPUT_ROOT  # noqa: E402
SECRET_CONFIG_PATH = notifier_mod.SECRET_CONFIG_PATH


load_secret_config = notifier_mod.load_secret_config
Notifier = notifier_mod.Notifier
TelegramNotifier = notifier_mod.TelegramNotifier
SlackNotifier = notifier_mod.SlackNotifier


def _secret_resolver(*args: Any) -> Any:
    if not args:
        raise ValueError("secret resolver expects a dotted key path")
    key = args[0]
    default = args[1] if len(args) > 1 else None
    return notifier_mod.get_secret_value(key, default=default)


if not OmegaConf.has_resolver("secret"):
    OmegaConf.register_new_resolver("secret", _secret_resolver, use_cache=True)

DEPENDENCY_HINTS: Dict[str, str] = {
    "psutil": "{exec} python3 -m pip install psutil",
    "pynvml": "{exec} python3 -m pip install nvidia-ml-py3",
    "tqdm": "{exec} python3 -m pip install tqdm",
}


def _labels_view(run_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical labels/inputs dict for a run."""
    if not run_cfg:
        return {}
    labels = run_cfg.get("labels")
    if labels is None:
        labels = run_cfg.get("meta") or run_cfg.get("combo") or run_cfg.get("inputs") or {}
    return labels or {}


def _normalize_gpu_ids(value: Any) -> List[str]:
    """Normalize CUDA_VISIBLE_DEVICES-style inputs into a list of GPU ids."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items: List[str] = []
        for item in value:
            items.extend(_normalize_gpu_ids(item))
        return items
    tokens = []
    if isinstance(value, str):
        tokens = re.split(r"[,\s]+", value.strip())
    else:
        tokens = [str(value)]
    cleaned: List[str] = []
    for token in tokens:
        token = token.strip()
        if not token or token.lower() in {"none", "null"}:
            continue
        if token not in cleaned:
            cleaned.append(token)
    return cleaned


def _detect_gpu_ids() -> List[str]:
    """Best-effort probe for available GPU ids using nvidia-smi."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    detected: List[str] = []
    for line in completed.stdout.splitlines():
        match = re.match(r"GPU\s+(\d+)", line.strip())
        if match:
            detected.append(match.group(1))
    return detected


def _gpu_pool(run_config: Optional["RunExecutionConfig"]) -> List[str]:
    """Determine the set of GPU ids to fan out across."""
    candidates: List[str] = []
    if run_config and run_config.gpu_ids:
        candidates = _normalize_gpu_ids(run_config.gpu_ids)
    if not candidates and run_config and run_config.env:
        candidates = _normalize_gpu_ids(run_config.env.get("CUDA_VISIBLE_DEVICES"))
    if not candidates:
        candidates = _normalize_gpu_ids(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if candidates:
        return candidates
    return _detect_gpu_ids()


def _component_output_base(component: str) -> Optional[Path]:
    """Best-effort resolution of the base output root for a component."""
    try:
        outputs = getattr(PATHS, "outputs", None)
        if outputs:
            if isinstance(outputs, dict) and component in outputs:
                return Path(str(outputs[component]))
            candidate = getattr(outputs, component, None)
            if candidate:
                return Path(str(candidate))
    except Exception:
        pass
    try:
        return Path(OUTPUT_ROOT) / component
    except Exception:
        return None


def _scope_output_paths(cfg: Dict[str, Any], component: str, run_id: str) -> Optional[Path]:
    """Rewrite output paths to a run-scoped directory to avoid clobbering."""
    base_root = _component_output_base(component)
    if base_root is None:
        return None
    scoped_root = base_root / "runs" / run_id
    base_str = str(base_root)
    scoped_str = str(scoped_root)

    def _rewrite(value: Any) -> Any:
        if isinstance(value, str) and base_str in value:
            return value.replace(base_str, scoped_str)
        return value

    for key in ("CMD", "PARAMS", "PREP_COMMANDS"):
        seq = cfg.get(key)
        if isinstance(seq, list):
            cfg[key] = [_rewrite(item) for item in seq]

    cfg["scoped_output_root"] = scoped_str
    return scoped_root


@dataclass
class ContainerRuntimeOptions:
    # Defaults are provided in Hydra configs; keep code defaults empty to avoid hidden divergence.
    mounts: List[str] = field(default_factory=list)
    env: List[str] = field(default_factory=list)
    extra_args: List[str] = field(default_factory=list)
    runner_entry: Tuple[str, ...] = field(
        default_factory=lambda: ("python3", "-m", "profile.cli.run_single_run")
    )

    def __post_init__(self) -> None:
        if not isinstance(self.runner_entry, tuple):
            self.runner_entry = tuple(self.runner_entry)


@dataclass
class FilterConfig:
    scaffolds: List[str] = field(default_factory=list)
    ligands: List[str] = field(default_factory=list)
    qualities: List[str] = field(default_factory=list)
    contigs: List[str] = field(default_factory=list)  # legacy alias

    def selected_scaffolds(self) -> List[str]:
        return self.scaffolds or self.contigs


@dataclass
class RunExecutionConfig:
    mps_enabled: bool = False
    concurrency: int = 0
    gpu_ids: Optional[List[str]] = None
    timeout_sec: Optional[int] = None
    env: Dict[str, str] = field(default_factory=dict)


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "profiling"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    mode: str = "online"


@dataclass
class SmokeConfig:
    timeout_sec: int = 5
    max_concurrency: int = 1
    report_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("PROFILE_STORAGE_ROOT", "/mnt/nfs/new/bioinformatics/profile")) / "raw" / "smoke"
    )
    count_timeout_as_success: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.report_dir, Path):
            self.report_dir = Path(self.report_dir)


@dataclass
class SmokeRecord:
    component: str
    run_id: str
    status: str
    start_time: str
    end_time: str
    meta: Dict[str, Any]
    profiler: Optional[str] = None
    exit_code: Optional[int] = None
    error_summary: Optional[str] = None


class SmokeReport:
    def __init__(self, *, report_root: Path, timestamp: str, profiler: Optional[str], count_timeout_as_success: bool = False) -> None:
        self.report_root = Path(report_root)
        self.timestamp = timestamp
        self.profiler = profiler
        self.count_timeout_as_success = bool(count_timeout_as_success)
        self.records: List[SmokeRecord] = []

    def add(self, record: SmokeRecord) -> None:
        self.records.append(record)

    def is_success(self, record: SmokeRecord) -> bool:
        if record.status == "success":
            return True
        if self.count_timeout_as_success and record.status == "timeout":
            return True
        return False

    def _as_dict(self, record: SmokeRecord) -> Dict[str, Any]:
        return {
            "component": record.component,
            "run_id": record.run_id,
            "status": record.status,
            "start_time": record.start_time,
            "end_time": record.end_time,
            "meta": record.meta,
            "profiler": record.profiler,
            "exit_code": record.exit_code,
            "error_summary": record.error_summary,
        }

    def write(self) -> Tuple[Path, Path]:
        root = self.report_root / self.timestamp
        root.mkdir(parents=True, exist_ok=True)
        json_path = root / "smoke_report.json"
        md_path = root / "smoke_report.md"

        payload = {
            "timestamp": self.timestamp,
            "profiler": self.profiler,
            "records": [self._as_dict(r) for r in self.records],
        }
        json_path.write_text(json.dumps(payload, indent=2))
        md_path.write_text(self._render_markdown())
        return json_path, md_path

    def _render_markdown(self) -> str:
        lines: List[str] = []
        lines.append(f"# Smoke Report ({self.timestamp})")
        if self.profiler:
            lines.append(f"- Profiler: {self.profiler}")
        component_summary: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for rec in self.records:
            component_summary[rec.component][rec.status] += 1
            component_summary[rec.component]["total"] += 1
        if component_summary:
            lines.append("\n## Component summary")
            for comp in sorted(component_summary):
                summary = component_summary[comp]
                successes = summary.get("success", 0)
                if self.count_timeout_as_success:
                    successes += summary.get("timeout", 0)
                timeouts = summary.get("timeout", 0)
                errors = summary.get("error", 0)
                total = summary.get("total", 0)
                lines.append(f"- {comp}: {successes}/{total} success, {timeouts} timeout, {errors} error")
        lines.append("\n## Runs")
        for rec in self.records:
            lines.append(f"- {rec.component} :: {rec.run_id} :: {rec.status}")
            if rec.error_summary:
                lines.append(f"  - error: {rec.error_summary}")
            if rec.status == "timeout" and self.count_timeout_as_success:
                lines.append("  - note: treated as success for smoke (hit timeout threshold)")
        return "\n".join(lines)


@dataclass
class ProfilingRunner:
    components: List[str] = field(default_factory=list)
    filters: FilterConfig = field(default_factory=FilterConfig)
    mode: str = "full"
    smoke: SmokeConfig = field(default_factory=SmokeConfig)
    repeats: int = 3
    profile_duration: int = 0
    run_config: RunExecutionConfig = field(default_factory=RunExecutionConfig)
    features: List[str] = field(default_factory=list)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    default_exec_helper: str = "docker/autodock_vina_gpu/exec.sh"
    container_runtime: ContainerRuntimeOptions = field(default_factory=ContainerRuntimeOptions)
    repo_root: Path = field(default_factory=lambda: REPO_ROOT)
    config_cache_dir: Path = field(default_factory=lambda: REPO_ROOT / ".hydra_cache")
    container_workdir: str = "/workspace"
    in_container: bool = False
    skip_existing: bool = True
    show_progress: bool = True
    dry_run: bool = False
    list_runs: bool = False
    ignore_nsight: bool = False
    timestamp: Optional[str] = None
    show_components: bool = False
    notifier: Optional[notifier_mod.Notifier] = None
    continue_on_error: bool = True
    experiments: Dict[str, Any] = field(default_factory=dict)

    _raw_cfg: Optional[DictConfig] = field(default=None, init=False, repr=False)
    _runner_cfg: Optional[DictConfig] = field(default=None, init=False, repr=False)
    _profiler_cfg: Optional[DictConfig] = field(default=None, init=False, repr=False)
    _component_cache: Dict[str, ComponentSpec] = field(default_factory=dict, init=False, repr=False)
    _start_time: Optional[datetime] = field(default=None, init=False, repr=False)
    _run_failures: List[SmokeRecord] = field(default_factory=list, init=False, repr=False)
    _profiler_override: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.repo_root, Path):
            self.repo_root = Path(self.repo_root)
        if not isinstance(self.config_cache_dir, Path):
            self.config_cache_dir = Path(self.config_cache_dir)
        if not self.config_cache_dir.is_absolute():
            self.config_cache_dir = (self.repo_root / self.config_cache_dir).resolve()
        self.mode = (self.mode or "full").lower()
        if self.mode not in {"full", "smoke"}:
            raise ValueError(f"runner.mode must be 'full' or 'smoke', got {self.mode}")

    def bind_config(self, cfg: DictConfig) -> None:
        self._raw_cfg = cfg
        self._runner_cfg = cfg.runner if "runner" in cfg else cfg
        self._profiler_cfg = cfg.get("profiler")
        try:
            from hydra.core.hydra_config import HydraConfig  # type: ignore

            overrides = getattr(HydraConfig.get(), "overrides", None)
            tasks = list(getattr(overrides, "task", []) or [])
            profiler_overrides = [item for item in tasks if str(item).startswith("profiler=")]
            if profiler_overrides:
                self._profiler_override = str(profiler_overrides[-1])
        except Exception:
            self._profiler_override = self._profiler_override or None
        if "repo_root" in cfg:
            self.repo_root = Path(cfg.repo_root)
        if "config_cache_dir" in cfg:
            path = Path(cfg.config_cache_dir)
            if not path.is_absolute():
                path = (self.repo_root / path).resolve()
            self.config_cache_dir = path
        if "experiments" in cfg:
            exp_cfg = OmegaConf.to_container(cfg.experiments, resolve=True) or {}
            # Some experiment configs nest under an "experiments" key; unwrap for convenience.
            if isinstance(exp_cfg, dict) and "experiments" in exp_cfg and isinstance(exp_cfg["experiments"], dict):
                exp_cfg = exp_cfg["experiments"]
            self.experiments = exp_cfg

    def __call__(self) -> int:
        return self.run()

    def run(self) -> int:
        configured = sorted(self.components or [])
        if self.show_components:
            print("Available components:")
            available = available_components()
            if not available:
                print("  (none discovered)")
            for comp in available:
                suffix = " (selected)" if comp in self.components else ""
                print(f"  - {comp}{suffix}")
            return 0

        experiment_components = list((self.experiments or {}).keys())
        selected = self.components or configured or experiment_components or available_components()
        known = set(available_components())
        unknown = sorted(comp for comp in selected if comp not in known)
        if unknown:
            print(
                "[WARN] Ignoring unknown component(s): "
                + ", ".join(unknown)
                + " (expect configs under profiling/configs/components)."
            )
            selected = [c for c in selected if c in known]
        if not selected:
            raise ValueError("No valid components selected. Add component configs or provide experiments/overrides.")

        if self._profiler_cfg is None:
            if self.list_runs or self.dry_run:
                self._profiler_cfg = OmegaConf.create({})
            else:
                raise RuntimeError("Profiler config is required; set `profiler=level_a|level_b|level_c`.")

        timestamp = self.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        if self._runner_cfg is not None:
            self._runner_cfg.timestamp = timestamp

        summary = ", ".join(selected)
        self._start_time = datetime.now()
        self._notify(
            f"[profiling] Starting sweep at {timestamp} (components: {summary or 'none'}, repeats={self.repeats})"
        )

        smoke_report: Optional[SmokeReport] = None
        profiler_name = _profiler_name(self._profiler_cfg)
        if self.mode == "smoke":
            smoke_report = SmokeReport(
                report_root=self.smoke.report_dir,
                timestamp=timestamp,
                profiler=profiler_name,
                count_timeout_as_success=self.smoke.count_timeout_as_success,
            )

        aggregated_runs: List[Tuple[ComponentSpec, List[Dict[str, Any]]]] = []
        global_progress = None
        total_planned_runs = 0

        for component in selected:
            try:
                spec = self._get_component_spec(component)
            except Exception as exc:
                print(f"[ERROR] Failed to load component '{component}': {exc}")
                self._record_run_result(
                    component=component,
                    run_id=f"{component}-init",
                    status="error",
                    start_time=datetime.utcnow().isoformat(),
                    end_time=datetime.utcnow().isoformat(),
                    meta={},
                    profiler=_profiler_name(self._profiler_cfg),
                    exit_code=None,
                    error_summary=str(exc),
                    smoke_report=smoke_report,
                )
                continue
            exp_combos = {}
            try:
                exp_combos = dict(self.experiments or {}).get(component, {})
            except Exception:
                exp_combos = dict(self.experiments) if isinstance(self.experiments, dict) else {}
                exp_combos = exp_combos.get(component, {})
            if getattr(spec, "batching", None) is not None and getattr(spec.batching, "output", None) is None:
                exp_combos = _collapse_output_samples(exp_combos, value=1)
            combos_override = _normalize_experiment_combos(exp_combos)
            def _call(builder, *, smoke: bool):
                kwargs = {
                    "timestamp": timestamp,
                    "repeats": self.repeats,
                    "profiler_label": profiler_name,
                }
                if combos_override is not None:
                    kwargs["combos"] = combos_override
                try:
                    return builder(**kwargs)
                except TypeError:
                    kwargs.pop("combos", None)
                    try:
                        return builder(**kwargs)
                    except TypeError:
                        kwargs.pop("profiler_label", None)
                        return builder(**kwargs)

            if self.mode == "smoke":
                runs = _call(spec.build_runs, smoke=True)
            else:
                runs = _call(spec.build_runs, smoke=False)
            if not runs:
                print(f"[WARN] No runs generated for component '{component}' (mode: {self.mode}).")
                continue
            runs = filter_runs(
                runs,
                scaffolds=self.filters.selected_scaffolds(),
                ligands=self.filters.ligands,
                qualities=self.filters.qualities,
            )

            if self.list_runs or self.dry_run:
                print(f"\n[{component}] matching runs: {len(runs)}")
                for cfg in runs:
                    meta = _labels_view(cfg)
                    util_csv = cfg.get("util_csv") or "(unknown)"
                    meta_desc = ", ".join(
                        f"{k}={v}" for k, v in meta.items() if v is not None
                    )
                    print("  - ", meta_desc, f"util_csv={util_csv}")
                if not self.dry_run:
                    continue

            aggregated_runs.append((spec, runs))
            total_planned_runs += len(runs)

        if aggregated_runs:
            if self.show_progress and not self.dry_run and total_planned_runs > 0:
                try:
                    from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel
                    global_progress = tqdm(total=total_planned_runs, desc="[sweep] runs", unit="run")
                except Exception:
                    global_progress = None
            try:
                for spec, runs in aggregated_runs:
                    try:
                        if smoke_report is not None:
                            run_component(spec, runs, self, smoke_report=smoke_report, progress=global_progress)
                        else:
                            run_component(spec, runs, self, progress=global_progress)
                    except Exception as exc:  # pragma: no cover
                        if smoke_report is None:
                            self._notify_failure(exc)
                            raise
                        print(f"[ERROR] Smoke run failed for component '{spec.name}': {exc}")
            finally:
                if global_progress:
                    global_progress.close()

        exit_code = 0
        if smoke_report is not None:
            json_path, md_path = smoke_report.write()
            print(f"\nSmoke report written to: {json_path} and {md_path}")
            has_failures = any(not smoke_report.is_success(rec) for rec in smoke_report.records)
            if has_failures:
                print("[INFO] Smoke tests completed with failures/timeouts. See report for details.")
                exit_code = 1
            else:
                timed_out = any(rec.status == "timeout" for rec in smoke_report.records)
                if timed_out and smoke_report.count_timeout_as_success:
                    print("[INFO] Smoke timeouts treated as success (count_timeout_as_success=true).")
        if self._run_failures:
            exit_code = exit_code or 1
            self._notify(f"[profiling] Sweep completed with {len(self._run_failures)} failure(s) in {self._elapsed_time()}")
        else:
            self._notify_success()
        if self.continue_on_error:
            exit_code = 0
        return exit_code

    def _get_component_spec(self, name: str) -> ComponentSpec:
        if name not in self._component_cache:
            self._component_cache[name] = load_component_spec(name)
        return self._component_cache[name]

    def run_runs_in_container(
        self,
        spec: ComponentSpec,
        runs: List[Dict[str, Any]],
        *args,
        smoke_report: Optional[SmokeReport] = None,
        progress: Any = None,
        **kwargs,
    ) -> None:
        if smoke_report is None and args:
            candidate = args[0]
            if isinstance(candidate, SmokeReport):
                smoke_report = candidate
            elif isinstance(candidate, bool):
                smoke_report = None if not candidate else SmokeReport(
                    report_root=self.smoke.report_dir,
                    timestamp=self.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"),
                    profiler=_profiler_name(self._profiler_cfg),
                    count_timeout_as_success=self.smoke.count_timeout_as_success,
                )
        if spec.container is None:
            raise RuntimeError(
                f"Component '{spec.name}' must define `container` for execution (host runtime unsupported)."
            )
        if not runs:
            print(f"[INFO] No runs selected for component '{spec.name}'.")
            return

        base_run_env = dict(self.run_config.env or {}) if self.run_config else {}
        gpu_ids = _gpu_pool(self.run_config)
        requested_workers = int(self.run_config.concurrency or 0) if self.run_config else 0
        if requested_workers <= 0:
            requested_workers = len(gpu_ids) if gpu_ids else 1
        worker_count = max(1, requested_workers)
        if gpu_ids:
            if worker_count > len(gpu_ids):
                print(
                    f"[INFO] Limiting concurrency to {len(gpu_ids)} GPU(s) "
                    f"(requested {worker_count})."
                )
            worker_count = min(worker_count, len(gpu_ids))
        component_limit = getattr(spec, "max_workers", None)
        try:
            component_limit = int(component_limit) if component_limit is not None else None
        except Exception:
            component_limit = None
        if component_limit and component_limit > 0:
            if worker_count > component_limit:
                print(
                    f"[INFO] Limiting concurrency for {spec.name} to "
                    f"{component_limit} worker(s) (max_workers)."
                )
            worker_count = min(worker_count, component_limit)
        assigned_gpu_ids = gpu_ids[:worker_count] if gpu_ids else [None] * worker_count

        local_progress = None
        if progress is None and self.show_progress and not self.dry_run:
            try:
                from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel
                local_progress = tqdm(total=len(runs), desc=f"[{spec.name}] runs", unit="run")
            except Exception:
                local_progress = None

        progress_lock = threading.Lock()
        result_lock = threading.Lock()
        abort_event = threading.Event()
        errors: List[BaseException] = []

        def _mark_progress() -> None:
            with progress_lock:
                if progress:
                    progress.update(1)
                elif local_progress:
                    local_progress.update(1)

        def _record_result_safe(**kwargs: Any) -> None:
            with result_lock:
                self._record_run_result(**kwargs)

        def _merge_run_env(gpu_id: Optional[str]) -> Dict[str, str]:
            env = dict(base_run_env)
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            return {k: v for k, v in env.items() if str(v) not in {"", "None", "null"}}

        def _container_gpus_arg(gpu_id: Optional[str]) -> Optional[str]:
            if gpu_id is None:
                return None
            return f"device={gpu_id}"

        run_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        for cfg in runs:
            run_queue.put(cfg)

        if gpu_ids:
            gpu_summary = ", ".join(assigned_gpu_ids)
            print(f"[{spec.name}] GPU pool: {gpu_summary} (workers={worker_count})")
        elif worker_count > 1:
            print(f"[{spec.name}] Using {worker_count} worker(s) without GPU affinity")

        def _launch_single(cfg: Dict[str, Any], gpu_id: Optional[str]) -> None:
            if abort_event.is_set():
                return
            run_id = self._format_run_id(spec, cfg)
            header = f"[{spec.name}] launching container for {run_id}"
            print("\n" + "=" * len(header))
            print(header)
            if gpu_id is not None:
                print(f"GPU: {gpu_id}")
            print("=" * len(header))

            cfg["run_id"] = run_id
            scoped_output_root = _scope_output_paths(cfg, spec.name, run_id)
            run_ctx = self._base_runner_context(run_id)
            if gpu_id is not None:
                run_ctx.setdefault("run_config", {})
                run_ctx["run_config"]["assigned_gpu"] = gpu_id
                cfg["assigned_gpu"] = gpu_id
            labels = _labels_view(cfg)
            if spec.container:
                container_payload = self._as_plain_payload(spec.container)
                run_ctx["container"] = container_payload
                cfg["container"] = container_payload
            if self.container_runtime:
                runtime_payload = self._as_plain_payload(self.container_runtime)
                run_ctx["container_runtime"] = runtime_payload
                cfg["container_runtime"] = runtime_payload
            if scoped_output_root is not None:
                run_ctx.setdefault("run_config", {})
                run_ctx["run_config"]["scoped_output_root"] = str(scoped_output_root)

            resolved_artifacts = _shared.resolve_profiler_artifacts(cfg, self._profiler_cfg, run_ctx)
            cfg["profiler_artifacts"] = [str(path) for path in resolved_artifacts]
            if not self.skip_existing and not self.dry_run:
                _reset_run_dirs(cfg)
            self._write_run_snapshot(cfg, run_ctx, self._profiler_cfg)
            profiler_ready = _should_skip_existing(resolved_artifacts, cfg)

            tracker = None
            if self.skip_existing and profiler_ready:
                print("[SKIP] Artifacts already exist; skipping container launch.")
                _mark_progress()
                return
            if not self.dry_run:
                tracker = _start_wandb_run(self.wandb_config(), spec.name, cfg, self._profiler_cfg, run_ctx)

            name_prefix = self._container_name_prefix(spec, run_id)
            base_runner_entry: Tuple[str, ...] = ("python3", "-m", "profile.cli.run_single_run")
            if spec.container and getattr(spec.container, "runner_entry", None):
                base_runner_entry = tuple(spec.container.runner_entry)

            payload_host: Optional[Path] = None
            payload_container: Optional[Path] = None
            if not self.dry_run:
                payload_host, payload_container = self._persist_run_payload(
                    spec,
                    cfg,
                    run_id,
                    include_wandb=False,
                    runner_ctx=run_ctx,
                )

            runner_entry = base_runner_entry
            if payload_container:
                runner_entry = runner_entry + ("--payload", str(payload_container))

            start_ts = datetime.utcnow().isoformat()
            status = "success"
            error_summary: Optional[str] = None
            exit_code: Optional[int] = 0
            capture_output = bool(smoke_report is not None or tracker is not None)
            captured_stdout: Optional[Any] = None
            captured_stderr: Optional[Any] = None
            fatal_exc: Optional[BaseException] = None
            run_env = _merge_run_env(gpu_id)
            gpus_arg = _container_gpus_arg(gpu_id)
            try:
                completed = run_component_in_container(
                    spec,
                    runner_entry=runner_entry,
                    name_prefix=name_prefix,
                    runtime=self.container_runtime,
                    run_env=run_env,
                    timeout=self.smoke.timeout_sec if smoke_report is not None else None,
                    capture_output=capture_output,
                    gpus_override=gpus_arg,
                )
                exit_code = getattr(completed, "returncode", 0)
                captured_stdout = getattr(completed, "stdout", None)
                captured_stderr = getattr(completed, "stderr", None)
            except subprocess.TimeoutExpired as exc:
                status = "timeout"
                error_summary = _summarize_process_error(exc)
                exit_code = None
                captured_stdout = getattr(exc, "stdout", None)
                captured_stderr = getattr(exc, "stderr", None)
                fatal_exc = exc if smoke_report is None and not self.continue_on_error else None
                if tracker:
                    tracker.log_output(stdout=getattr(exc, "stdout", None), stderr=getattr(exc, "stderr", None))
            except subprocess.CalledProcessError as exc:
                status = "timeout" if exc.returncode == 124 else "error"
                exit_code = exc.returncode
                error_summary = _summarize_process_error(exc)
                captured_stdout = getattr(exc, "stdout", None)
                captured_stderr = getattr(exc, "stderr", None)
                fatal_exc = exc if smoke_report is None and not self.continue_on_error else None
                if getattr(exc, "stdout", None):
                    print(exc.stdout)
                if getattr(exc, "stderr", None):
                    print(exc.stderr)
                if tracker:
                    tracker.log_output(stdout=getattr(exc, "stdout", None), stderr=getattr(exc, "stderr", None))
            except Exception as exc:  # pragma: no cover - defensive catchall
                status = "error"
                error_summary = str(exc)
                exit_code = None
                captured_stdout = getattr(exc, "stdout", None)
                captured_stderr = getattr(exc, "stderr", None)
                fatal_exc = exc if smoke_report is None and not self.continue_on_error else None
                if tracker:
                    tracker.log_output(stdout=None, stderr=str(exc))
            except BaseException as exc:  # pragma: no cover - fatal signals/SystemExit
                status = "error"
                error_summary = str(exc)
                exit_code = None
                captured_stdout = getattr(exc, "stdout", None)
                captured_stderr = getattr(exc, "stderr", None)
                fatal_exc = exc
                if tracker:
                    tracker.log_output(stdout=None, stderr=str(exc))
            finally:
                _mark_progress()
                if payload_host and payload_host.exists() and self.dry_run:
                    payload_host.unlink(missing_ok=True)

            end_ts = datetime.utcnow().isoformat()
            if tracker:
                if status == "success":
                    tracker.log_output(stdout=captured_stdout, stderr=captured_stderr)
                tracker.finish(
                    status="completed" if status == "success" else status,
                    artifacts=resolved_artifacts if status == "success" else None,
                    error=error_summary,
                    stdout=captured_stdout,
                    stderr=captured_stderr,
                )
            if fatal_exc:
                with result_lock:
                    errors.append(fatal_exc)
                    abort_event.set()
                return
            if smoke_report is not None or status != "success":
                _record_result_safe(
                    component=spec.name,
                    run_id=run_id,
                    status=status,
                    start_time=start_ts,
                    end_time=end_ts,
                    meta=labels,
                    profiler=_profiler_name(self._profiler_cfg),
                    exit_code=exit_code,
                    error_summary=error_summary,
                    smoke_report=smoke_report,
                )

        def _worker(gpu_id: Optional[str]) -> None:
            while not abort_event.is_set():
                try:
                    cfg = run_queue.get_nowait()
                except queue.Empty:
                    return
                try:
                    if abort_event.is_set():
                        return
                    _launch_single(cfg, gpu_id)
                finally:
                    run_queue.task_done()

        threads: List[threading.Thread] = []
        for gpu_id in assigned_gpu_ids:
            thread = threading.Thread(target=_worker, args=(gpu_id,), daemon=True)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        if local_progress:
            local_progress.close()

        should_raise = False
        if errors:
            has_non_exception = any(not isinstance(exc, Exception) for exc in errors)
            if has_non_exception:
                should_raise = True
            elif smoke_report is None and not self.continue_on_error:
                should_raise = True
        if should_raise:
            raise errors[0]

    def _build_cli_overrides(self, spec: ComponentSpec, cfg: Dict[str, Any], run_id: str) -> List[str]:
        overrides: List[str] = []
        if self._profiler_override:
            overrides.append(self._profiler_override)
        overrides.append(f"runner.repeats={self.repeats}")
        overrides.append(f"runner.mode={self.mode}")
        overrides.append(f"runner.skip_existing={str(self.skip_existing).lower()}")
        overrides.append(f"runner.ignore_nsight={str(self.ignore_nsight).lower()}")
        if self.timestamp:
            overrides.append(f"runner.timestamp={self.timestamp}")
        if self.repo_root:
            overrides.append(f"repo_root={self.repo_root}")
        if self.features:
            feature_list = ",".join(json.dumps(str(item)) for item in self.features if item is not None)
            overrides.append(f"runner.features=[{feature_list}]")
        overrides.append(f"runner.components=[{spec.name}]")

        labels = dict(_labels_view(cfg))
        for key, value in labels.items():
            if value is None:
                continue
            overrides.append(f"+run.{key}={value}")
        return overrides

    def _format_run_id(self, spec: ComponentSpec, cfg: Dict[str, Any]) -> str:
        meta = _labels_view(cfg)
        parts = [spec.name]
        sample_id = meta.get("sample_id")
        if sample_id:
            parts.append(str(sample_id))
        quality = meta.get("quality")
        if quality:
            parts.append(str(quality))
        input_bs = meta.get("input_batch_size")
        if input_bs is not None:
            parts.append(f"inp{input_bs}")
        output_samples = meta.get("output_samples")
        if output_samples is not None:
            parts.append(f"out{output_samples}")
        repeat_idx = (
            cfg.get("repeat_idx")
            or meta.get("repeat_idx")
            or cfg.get("run_idx")
        )
        if repeat_idx:
            parts.append(f"r{repeat_idx}")
        return "-".join(str(part) for part in parts if part)

    def _as_plain_tree(self, value: Any) -> Any:
        if isinstance(value, DictConfig):
            # Avoid forcing resolution so unresolved interpolations (e.g., ${runner.run_id}) don't explode.
            value = OmegaConf.to_container(value, resolve=False)
        if is_dataclass(value):
            value = asdict(value)
        if isinstance(value, dict):
            return {k: self._as_plain_tree(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._as_plain_tree(v) for v in value]
        if isinstance(value, tuple):
            return [self._as_plain_tree(v) for v in value]
        if isinstance(value, set):
            return [self._as_plain_tree(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        return value

    def _write_run_snapshot(
        self,
        run_cfg: Dict[str, Any],
        runner_ctx: Dict[str, Any],
        profiler_cfg: Optional[Any],
    ) -> None:
        base_dir = Path(run_cfg.get("PROF_DATA_DIR") or self.config_cache_dir)
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            snapshot_raw = {
                "run": self._as_plain_tree(run_cfg),
                "runner_ctx": self._as_plain_tree(runner_ctx),
                "profiler_cfg": self._as_plain_tree(profiler_cfg),
            }
            # Ensure YAML-safe payload (no custom objects).
            snapshot = json.loads(json.dumps(snapshot_raw, default=str))
            snapshot_path = base_dir / "config.yaml"
            snapshot_path.write_text(yaml.safe_dump(snapshot, sort_keys=False))
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[WARN] Failed to write run snapshot to {base_dir}: {exc}")

    def _container_name_prefix(self, spec: ComponentSpec, run_id: str) -> Optional[str]:
        if spec.container is None:
            return None
        base = spec.container.name_prefix or "profile-run"
        suffix = self._slugify(run_id, max_length=32)
        return f"{base}-{suffix}"

    def _to_container_path(self, host_path: Path) -> Path:
        base_root = Path(self.repo_root).resolve()
        try:
            relative = host_path.resolve().relative_to(base_root)
        except ValueError:
            relative = Path(host_path.name)
        return Path(self.container_workdir) / relative

    def _base_runner_context(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        runner_cfg = None
        try:
            if isinstance(self._runner_cfg, DictConfig):
                runner_cfg = OmegaConf.to_container(self._runner_cfg, resolve=True)
        except Exception:
            runner_cfg = None
        container_runtime = self._as_plain_payload(self.container_runtime) if self.container_runtime else None
        features = [str(feature) for feature in (self.features or []) if feature is not None]
        runner_node: Dict[str, Any] = {
            "profile_duration": self.profile_duration,
            "features": features,
        }
        if run_id:
            runner_node["run_id"] = run_id
        context: Dict[str, Any] = {
            "runner": runner_node,
            "run_config": self._as_plain_payload(self.run_config) or {},
            "features": features,
        }
        if runner_cfg is not None:
            context["runner_cfg"] = runner_cfg
        if container_runtime is not None:
            context["container_runtime"] = container_runtime
        if self._profiler_cfg is not None:
            profiler_context = _shared._build_profiler_runtime_context({}, context)
            prepared_profiler_cfg = _shared._prepare_profiler_cfg(self._profiler_cfg, profiler_context)
            context["profiler_cfg"] = OmegaConf.to_container(
                prepared_profiler_cfg,
                resolve=bool(run_id),
            )
        return context

    def _persist_run_payload(
        self,
        spec: ComponentSpec,
        run_cfg: Dict[str, Any],
        run_id: str,
        include_wandb: bool = True,
        include_profiler: bool = True,
        runner_ctx: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, Path]:
        cache_dir = Path(self.config_cache_dir or self.repo_root / ".hydra_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        slug = self._slugify(run_id or spec.name, max_length=80)
        filename = f"{spec.name}_{slug}.json"
        payload_path = cache_dir / filename

        # Prepare context that will also be embedded for container runs.
        run_context = runner_ctx or self._base_runner_context(run_id)
        if spec.container:
            container_payload = self._as_plain_payload(spec.container)
            run_context["container"] = container_payload
            run_cfg["container"] = container_payload
        if self.container_runtime:
            runtime_payload = self._as_plain_payload(self.container_runtime)
            run_context["container_runtime"] = runtime_payload
            run_cfg["container_runtime"] = runtime_payload

        profiler_cfg = self.profiler_config()
        resolved_artifacts = _shared.resolve_profiler_artifacts(run_cfg, profiler_cfg, run_context)
        run_cfg["profiler_artifacts"] = [str(path) for path in resolved_artifacts]

        payload: Dict[str, Any] = {
            "component": spec.name,
            "run_id": run_id,
            "run_cfg": run_cfg,
            "runner_ctx": run_context,
            "profiler_cfg": self._serialize_profiler_cfg() if include_profiler else None,
            "wandb": asdict(self.wandb_config()) if include_wandb else None,
            "exec_helper": getattr(spec, "exec_helper", None) or self.default_exec_helper,
            "skip_existing": bool(self.skip_existing),
            "ignore_nsight": bool(self.ignore_nsight),
        }

        payload_path.write_text(json.dumps(payload, indent=2, default=str))
        container_path = self._to_container_path(payload_path)

        # Also persist a resolved config snapshot alongside run artifacts for later inspection.
        self._write_run_snapshot(run_cfg, run_context, self._profiler_cfg)
        return payload_path, container_path

    def _serialize_profiler_cfg(self) -> Optional[Dict[str, Any]]:
        if self._profiler_cfg is None:
            return None
        return OmegaConf.to_container(self._profiler_cfg, resolve=False)

    @staticmethod
    def _as_plain_payload(value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, DictConfig):
            return OmegaConf.to_container(value, resolve=True)
        if isinstance(value, list):
            return [ProfilingRunner._as_plain_payload(item) for item in value]
        return value

    def profiler_config(self) -> Optional[DictConfig]:
        return self._profiler_cfg

    def wandb_config(self) -> WandbConfig:
        return self.wandb

    @staticmethod
    def _slugify(value: str, *, max_length: int = 32) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
        if not slug:
            slug = "run"
        if len(slug) > max_length:
            slug = slug[:max_length].rstrip("-")
        return slug or "run"

    def _notify(self, message: str) -> None:
        if self.notifier:
            try:
                self.notifier.send(message)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[WARN] Failed to send notification: {exc}")

    def _notify_failure(self, exc: Exception) -> None:
        elapsed = self._elapsed_time()
        self._notify(f"[profiling] Sweep failed after {elapsed}: {exc}")

    def _notify_success(self) -> None:
        elapsed = self._elapsed_time()
        self._notify(f"[profiling] Sweep completed in {elapsed}")

    def _elapsed_time(self) -> str:
        if not self._start_time:
            return "0s"
        delta = datetime.now() - self._start_time
        seconds = int(delta.total_seconds())
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        parts = []
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")
        return " ".join(parts)

    def _record_run_result(
        self,
        *,
        component: str,
        run_id: str,
        status: str,
        start_time: str,
        end_time: str,
        meta: Dict[str, Any],
        profiler: Optional[str],
        exit_code: Optional[int],
        error_summary: Optional[str],
        smoke_report: Optional[SmokeReport],
    ) -> None:
        record = SmokeRecord(
            component=component,
            run_id=run_id,
            status=status,
            start_time=start_time,
            end_time=end_time,
            meta=meta,
            profiler=profiler,
            exit_code=exit_code,
            error_summary=error_summary,
        )
        is_success = record.status == "success"
        if smoke_report is not None:
            smoke_report.add(record)
            is_success = smoke_report.is_success(record)
        if not is_success:
            self._run_failures.append(record)


def _ensure_path(value: Optional[str]) -> Optional[Path]:
    return Path(value) if value else None


# ... rest of helper functions copied from previous version (system artifacts, ensure_runtime_dependencies, etc.)

def _profiler_artifacts(cfg: Dict[str, Any]) -> List[Path]:
    explicit = cfg.get("profiler_artifacts")
    if explicit:
        return [Path(path) for path in explicit]
    basename = cfg.get("BASENAME")
    prof_dir = _ensure_path(cfg.get("PROF_DATA_DIR"))
    if not basename or prof_dir is None:
        return []
    return [
        prof_dir / f"{basename}_kernel_summary.csv",
        prof_dir / f"{basename}_ncu_summary.csv",
    ]


def _profiler_requires_psutil(profiler_cfg: Optional[Any]) -> bool:
    if profiler_cfg is None:
        return False
    target = None
    if isinstance(profiler_cfg, DictConfig):
        target = profiler_cfg.get("_target_")
    elif isinstance(profiler_cfg, dict):
        target = profiler_cfg.get("_target_")
    else:
        target_cls = getattr(profiler_cfg, "__class__", None)
        if target_cls:
            target = f"{target_cls.__module__}.{target_cls.__name__}"
    if isinstance(target, str):
        return "LevelATelemetryStrategy" in target or target.endswith("TelemetryStrategy")
    return False


def _profiler_requires_nsight(profiler_cfg: Optional[Any]) -> bool:
    if profiler_cfg is None:
        return True
    value = profiler_cfg.get("requires_nsight")
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _profiler_name(profiler_cfg: Optional[Any]) -> Optional[str]:
    if profiler_cfg is None:
        return None
    if isinstance(profiler_cfg, DictConfig):
        return profiler_cfg.get("name")
    if isinstance(profiler_cfg, dict):
        return profiler_cfg.get("name")
    return getattr(profiler_cfg, "name", None)


def _artifact_status(paths: List[Path]) -> Dict[str, Any]:
    present: List[Path] = []
    missing: List[Path] = []
    for path in paths:
        if path.exists() and path.is_file():
            try:
                if path.stat().st_size > 0:
                    present.append(path)
                    continue
            except OSError:
                pass
        missing.append(path)
    ready = bool(paths) and not missing
    return {"ready": ready, "present": present, "missing": missing}


def _candidate_marker_paths(base_path: Path) -> List[Path]:
    """Generate legacy marker locations by stripping batch/timestamp buckets."""
    candidates: List[Path] = []
    anchor = base_path.anchor
    parts = list(base_path.parts)
    if anchor:
        parts = parts[1:]

    def _add(parts_slice: List[str]) -> None:
        path = Path(anchor, *parts_slice) if anchor else Path(*parts_slice)
        if path not in candidates:
            candidates.append(path)

    _add(parts)

    without_batch = [p for p in parts if not re.match(r"(input|output)_\d+$", p)]
    if len(without_batch) != len(parts):
        _add(without_batch)

    without_ts = [p for p in without_batch if not re.match(r"\d{8}_\d{6}$", p)]
    if len(without_ts) != len(without_batch):
        _add(without_ts)

    return candidates


def _marker_status(cfg: Dict[str, Any]) -> Optional[str]:
    """Best-effort status for a previous run: success/error/timeout/... or None if absent.

    Only consult markers/status under PROF_DATA_DIR; raw directories are the canonical store.
    """

    def _check_path(run_path: Path) -> Optional[str]:
        status_json = run_path / "profile_status.json"
        if status_json.exists() and status_json.is_file():
            try:
                data = json.loads(status_json.read_text())
                status = str(data.get("status", "")).lower()
                return status or None
            except Exception:
                pass
        for marker, value in (
            ("profile_SUCCESS", "success"),
            ("profile_ERROR", "error"),
            ("profile_TIMEOUT", "timeout"),
        ):
            marker_path = run_path / marker
            if marker_path.exists() and marker_path.is_file():
                return value
        return None

    raw_dir = cfg.get("PROF_DATA_DIR")
    if raw_dir:
        for candidate in _candidate_marker_paths(Path(raw_dir)):
            status = _check_path(candidate)
            if status is not None:
                return status
    return None


def _should_skip_existing(artifacts: List[Path], cfg: Dict[str, Any]) -> bool:
    """Return True when a previous run completed successfully.

    Prefer marker files when present; fall back to artifact presence for legacy runs.
    """
    status = _marker_status(cfg)
    if status is not None:
        return status == "success"
    return _artifact_status(artifacts)["ready"]


def _marker_ready(cfg: Dict[str, Any]) -> bool:
    """Check marker files recorded by profiler strategies (profile_SUCCESS/status.json).

    Canonical location is the raw profiler directory (PROF_DATA_DIR).
    """
    def _check_path(run_path: Path) -> Optional[bool]:
        status_json = run_path / "profile_status.json"
        if status_json.exists() and status_json.is_file():
            try:
                data = json.loads(status_json.read_text())
                status = str(data.get("status", "")).lower()
                if status == "success":
                    return True
                if status:
                    marker = run_path / "profile_SUCCESS"
                    try:
                        if marker.exists():
                            marker.unlink()
                    except OSError:
                        pass
                    return False
            except Exception:
                pass
        marker = run_path / "profile_SUCCESS"
        if marker.exists() and marker.is_file():
            return True
        return None

    raw_dir = cfg.get("PROF_DATA_DIR")
    if raw_dir:
        for candidate in _candidate_marker_paths(Path(raw_dir)):
            result = _check_path(candidate)
            if result is not None:
                return result
        return False
    return False


def _reset_run_dirs(cfg: Dict[str, Any]) -> None:
    """Delete per-run profiler directories to avoid stale markers on reruns."""
    seen = set()
    for key in ("PROF_DIR", "PROF_DATA_DIR"):
        path = cfg.get(key)
        if not path:
            continue
        if path in seen:
            continue
        seen.add(path)
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception as exc:  # pragma: no cover - best effort cleanup
            print(f"[WARN] Failed to clear existing run directory {path}: {exc}")


def _format_paths(paths: List[Path]) -> str:
    if not paths:
        return "(none)"
    return ", ".join(str(path) for path in paths)


def _tail_output(text: Optional[Any], *, lines: int = 15) -> Optional[str]:
    if text is None:
        return None
    if isinstance(text, (bytes, bytearray)):
        text = text.decode("utf-8", errors="replace")
    elif not isinstance(text, str):
        text = str(text)
    if not text:
        return None
    chunks = text.strip().splitlines()
    if not chunks:
        return None
    return "\n".join(chunks[-lines:])


def _summarize_process_error(exc: Exception) -> str:
    """Render a short summary from subprocess output (stdout/stderr tails)."""
    parts: List[str] = []
    stdout = getattr(exc, "stdout", None)
    stderr = getattr(exc, "stderr", None)
    tail_stdout = _tail_output(stdout)
    tail_stderr = _tail_output(stderr)
    if tail_stdout:
        parts.append("STDOUT:\n" + tail_stdout)
    if tail_stderr:
        parts.append("STDERR:\n" + tail_stderr)
    if not parts:
        parts.append(str(exc))
    return "\n".join(parts)


def available_components() -> List[str]:
    return list_component_names()


def _normalize_experiment_combos(raw: Any) -> Optional[List[Dict[str, Any]]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        has_list = any(isinstance(v, list) for v in raw.values())
        if has_list:
            return list(_shared.iter_combos(raw))
        return [raw]
    return None


def _collapse_output_samples(raw: Any, *, value: int = 1) -> Any:
    """Collapse output_samples sweeps for components without output fan-out."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        if "output_samples" not in raw:
            return raw
        collapsed = dict(raw)
        current = collapsed.get("output_samples")
        if isinstance(current, list):
            collapsed["output_samples"] = [value]
        else:
            collapsed["output_samples"] = value
        return collapsed
    if isinstance(raw, list):
        normalized: List[Any] = []
        for item in raw:
            if isinstance(item, dict):
                collapsed = dict(item)
                if "output_samples" in collapsed:
                    collapsed["output_samples"] = value
                normalized.append(collapsed)
            else:
                normalized.append(item)
        seen: set[str] = set()
        deduped: List[Any] = []
        for item in normalized:
            if not isinstance(item, dict):
                deduped.append(item)
                continue
            key = json.dumps(item, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped
    return raw


def ensure_requirements(
    profiler_cfg: Optional[Any],
    *,
    need_progress: bool,
    exec_helper: Optional[str] = None,
    ignore_nsight: bool = False,
) -> None:
    """Centralized dependency checks for profiler runs."""
    required = set()
    if _profiler_requires_psutil(profiler_cfg):
        required.add("psutil")
    if need_progress:
        required.add("tqdm")

    missing = []
    for module_name in sorted(required):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            hint = DEPENDENCY_HINTS.get(module_name)
            if hint and "{exec}" in hint:
                helper = exec_helper or "exec.sh"
                hint = hint.format(exec=helper)
            missing.append((module_name, hint))

    if missing:
        print("[ERROR] Missing Python dependencies required for profiling runs:")
        for module_name, hint in missing:
            if hint:
                print(f"  - {module_name}: run `{hint}`")
            else:
                print(f"  - {module_name}")
        raise SystemExit(1)

    if _profiler_requires_nsight(profiler_cfg):
        ensure_nsight_tools(ignore_missing=ignore_nsight)


# Backward compatibility for any external callers.
def ensure_runtime_dependencies(profiler_cfg: Optional[Any], need_progress: bool,
                                exec_helper: Optional[str] = None) -> None:
    ensure_requirements(profiler_cfg, need_progress=need_progress, exec_helper=exec_helper)


def filter_runs(runs: List[Dict[str, Any]], *, scaffolds: Optional[Iterable[str]] = None,
                ligands: Optional[Iterable[str]] = None, qualities: Optional[Iterable[str]] = None,
                contigs: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    scaffolds_iter: Iterable[str]
    if scaffolds:
        scaffolds_iter = scaffolds
    elif contigs:
        scaffolds_iter = contigs
    else:
        scaffolds_iter = []

    scaffold_set = set(scaffolds_iter)
    ligand_set = set(ligands or [])
    quality_set = set(qualities or [])

    def match(meta: Dict[str, Any]) -> bool:
        scaffold_value = meta.get("scaffold") or meta.get("contig")
        if scaffold_set and scaffold_value not in scaffold_set:
            return False
        if ligand_set and meta.get("ligand") not in ligand_set:
            return False
        if quality_set and meta.get("quality") not in quality_set:
            return False
        return True

    return [run for run in runs if match(_labels_view(run))]


NSIGHT_INSTALL_HINT = "/workspace/docker/tools/install_nsight_tools.sh"


def ensure_nsight_tools(ignore_missing: bool = False) -> None:
    missing = [tool for tool in ("nsys", "ncu") if shutil.which(tool) is None]
    if missing and not ignore_missing:
        hint = NSIGHT_INSTALL_HINT
        raise RuntimeError(
            "Missing required Nsight tools: {}. Inside the profiling container, run: {}".format(
                ", ".join(missing), hint
            )
        )
    if missing:
        print(
            f"[WARN] Nsight tool(s) missing: {', '.join(missing)}; "
            f"install via {NSIGHT_INSTALL_HINT} if you need GPU profiling."
        )


def _merge_tags(*sources: Any) -> List[str]:
    seen = set()
    tags: List[str] = []
    for source in sources:
        if not source:
            continue
        if isinstance(source, str):
            values = [source]
        elif isinstance(source, dict):
            values = [source]
        elif isinstance(source, Iterable):
            values = source
        else:
            values = [source]
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            tags.append(text)
    return tags


def _collect_wandb_features(cfg: Optional[WandbConfig], run_cfg: Dict[str, Any], runner_ctx: Dict[str, Any]) -> List[str]:
    seen = set()
    features: List[str] = []

    def _extend(source: Any) -> None:
        if not source:
            return
        if isinstance(source, str):
            values = [source]
        elif isinstance(source, dict):
            values = [source]
        elif isinstance(source, Iterable):
            values = source
        else:
            values = [source]
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            features.append(text)

    _extend(getattr(cfg, "features", None))
    _extend(run_cfg.get("features"))
    labels = _labels_view(run_cfg)
    _extend(labels.get("features"))
    _extend(runner_ctx.get("features"))
    _extend((runner_ctx.get("runner") or {}).get("features"))
    return features


def _make_wandb_settings(module: Any) -> Any:
    settings_cls = getattr(module, "Settings", None)
    if settings_cls is None:
        return None
    for kwargs in (
        {"console": "redirect", "capture_stdout": True, "capture_stderr": True},
        {"console": "redirect"},
    ):
        try:
            return settings_cls(**kwargs)
        except Exception:
            continue
    return None


_WANDB_STATE_LOCK = threading.Lock()
_WANDB_DISABLED_REASON: Optional[str] = None
_WANDB_DISABLED_WARNING_EMITTED = False


def _describe_invalid_wandb(module: Any) -> str:
    origin = getattr(module, "__file__", None)
    detail = f" ({origin})" if origin else ""
    local_hint = ""
    candidate_dirs = {Path.cwd() / "wandb", REPO_ROOT / "wandb"}
    for candidate in candidate_dirs:
        if candidate.is_dir() and not (candidate / "__init__.py").exists():
            local_hint = f"; found local directory '{candidate}' shadowing the W&B package"
            break
    return f"imported `wandb` but it has no callable `init`{detail}{local_hint}"


def _disable_wandb(reason: str) -> None:
    global _WANDB_DISABLED_REASON
    with _WANDB_STATE_LOCK:
        _WANDB_DISABLED_REASON = reason


def _warn_wandb_disabled(reason: str) -> None:
    global _WANDB_DISABLED_WARNING_EMITTED
    with _WANDB_STATE_LOCK:
        if _WANDB_DISABLED_WARNING_EMITTED:
            return
        _WANDB_DISABLED_WARNING_EMITTED = True
    print(f"[WARN] W&B logging disabled: {reason}")


def _resolve_wandb_module() -> Tuple[Optional[Any], Optional[str]]:
    global _WANDB_DISABLED_REASON
    with _WANDB_STATE_LOCK:
        module = sys.modules.get("wandb")
        init_fn = getattr(module, "init", None) if module is not None else None
        if callable(init_fn):
            _WANDB_DISABLED_REASON = None
            return module, None
        if _WANDB_DISABLED_REASON is not None:
            return None, _WANDB_DISABLED_REASON

        try:
            module = importlib.import_module("wandb")
        except ModuleNotFoundError:
            _WANDB_DISABLED_REASON = (
                "wandb is not installed (install with `pip install wandb` or set `runner.wandb.enabled=false`)"
            )
            return None, _WANDB_DISABLED_REASON

        init_fn = getattr(module, "init", None)
        if not callable(init_fn):
            _WANDB_DISABLED_REASON = _describe_invalid_wandb(module)
            return None, _WANDB_DISABLED_REASON
        _WANDB_DISABLED_REASON = None
        return module, None


def _start_wandb_run(
    cfg: Any,
    component: str,
    run_cfg: Dict[str, Any],
    profiler_cfg: Optional[Any],
    runner_ctx: Dict[str, Any],
) -> Optional["WandbRunTracker"]:
    normalized = _normalize_wandb_cfg(cfg)
    if not normalized or not normalized.enabled:
        return None

    wandb, import_error = _resolve_wandb_module()
    if wandb is None:
        if import_error:
            _warn_wandb_disabled(import_error)
        return None

    run_id = run_cfg.get("run_id") or run_cfg.get("BASENAME") or component
    labels = dict(_labels_view(run_cfg))
    meta = labels
    combo = labels
    features = _collect_wandb_features(normalized, run_cfg, runner_ctx)
    profiler_name = None
    if profiler_cfg is not None:
        if isinstance(profiler_cfg, DictConfig):
            profiler_name = profiler_cfg.get("name")
        elif isinstance(profiler_cfg, dict):
            profiler_name = profiler_cfg.get("name")
        else:
            profiler_name = getattr(profiler_cfg, "name", None)

    config_payload = {
        "component": component,
        "run_id": run_id,
        "combo": combo,
        "meta": meta,
        "labels": labels,
        "profiler": profiler_name,
        "profiler_requires_nsight": _profiler_requires_nsight(profiler_cfg),
        "run_config": runner_ctx.get("run_config", {}),
        "features": features,
    }
    container_cfg = run_cfg.get("container") or runner_ctx.get("container")
    if container_cfg:
        config_payload["container"] = ProfilingRunner._as_plain_payload(container_cfg)
    container_runtime = runner_ctx.get("container_runtime")
    if container_runtime:
        config_payload["container_runtime"] = ProfilingRunner._as_plain_payload(container_runtime)
    runner_cfg = runner_ctx.get("runner_cfg")
    if runner_cfg:
        config_payload["runner_cfg"] = runner_cfg
    if profiler_cfg is not None:
        profiler_context = _shared._build_profiler_runtime_context(run_cfg, runner_ctx or {})
        prepared_profiler_cfg = _shared._prepare_profiler_cfg(profiler_cfg, profiler_context)
        config_payload["profiler_cfg"] = ProfilingRunner._as_plain_payload(prepared_profiler_cfg)

    feature_tags = [f"feature:{feat}" for feat in features]
    tags = _merge_tags(normalized.tags, [component], features, feature_tags)
    quality = labels.get("quality")
    if quality:
        tags = _merge_tags(tags, [quality])
    quality_level = labels.get("quality_level")
    if quality_level:
        tags = _merge_tags(tags, [f"quality_level:{quality_level}"])

    settings = _make_wandb_settings(wandb)
    init_kwargs = {
        "project": normalized.project,
        "entity": normalized.entity,
        "config": config_payload,
        "tags": tags,
        "mode": normalized.mode,
        "reinit": True,
        "group": component,
        "name": run_id,
    }
    if settings is not None:
        init_kwargs["settings"] = settings

    if "WANDB_DIR" not in os.environ:
        wandb_root = REPO_ROOT / ".wandb"
        try:
            wandb_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            wandb_root = None
        if wandb_root is not None:
            init_kwargs.setdefault("dir", str(wandb_root))

    try:
        handle = wandb.init(**init_kwargs)
    except Exception as exc:
        reason = f"wandb.init failed: {exc}"
        _disable_wandb(reason)
        _warn_wandb_disabled(reason)
        return None
    return WandbRunTracker(handle, run_id)


def _normalize_wandb_cfg(cfg: Any) -> Optional[WandbConfig]:
    if cfg is None:
        return None
    if isinstance(cfg, WandbConfig):
        return cfg
    if isinstance(cfg, DictConfig):
        data = OmegaConf.to_container(cfg, resolve=True)
    elif isinstance(cfg, dict):
        data = cfg
    else:
        return None
    return WandbConfig(**data)


class WandbRunTracker:
    """Lightweight wrapper to record profiling run status in Weights & Biases."""

    def __init__(self, run_handle: Any, run_id: str) -> None:
        self._run = run_handle
        self.run_id = run_id
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="wandb_logs_"))
        self._console_logged = False

    @staticmethod
    def _stringify_output(output: Optional[Any]) -> Optional[str]:
        if output is None:
            return None
        if isinstance(output, (bytes, bytearray)):
            return output.decode("utf-8", errors="replace")
        return str(output)

    def _emit_console_logs(self, *, stdout_tail: Optional[str], stderr_tail: Optional[str]) -> None:
        """Send short output snippets to W&B logs (or stdout/stderr fallback)."""
        try:
            import importlib

            wandb = importlib.import_module("wandb")
        except Exception:
            wandb = None

        def emit(label: str, text: Optional[str], *, err: bool = False) -> None:
            if not text:
                return
            sink = getattr(self._run, "termerror" if err else "termlog", None)
            if sink is None and wandb is not None:
                sink = getattr(wandb, "termerror" if err else "termlog", None)
            stream = sys.stderr if err else sys.stdout
            for line in text.splitlines():
                message = f"[{self.run_id} {label}] {line}"
                if callable(sink):
                    try:
                        sink(message)
                        continue
                    except Exception:
                        sink = None
                try:
                    print(message, file=stream)
                except Exception:
                    break

        emit("stdout", stdout_tail, err=False)
        emit("stderr", stderr_tail, err=True)

    def log_output(self, *, stdout: Optional[Any], stderr: Optional[Any], tail_lines: int = 200) -> None:
        if not self._run:
            return
        payload: Dict[str, Any] = {}
        tail_stdout = _tail_output(stdout, lines=tail_lines)
        tail_stderr = _tail_output(stderr, lines=tail_lines)
        if tail_stdout:
            payload["stdout"] = tail_stdout
        if tail_stderr:
            payload["stderr"] = tail_stderr
        if payload:
            try:
                self._run.log(payload)
            except Exception:
                pass
        self._emit_console_logs(stdout_tail=tail_stdout, stderr_tail=tail_stderr)
        if tail_stdout or tail_stderr:
            self._console_logged = True

    def finish(
        self,
        *,
        status: str,
        artifacts: Optional[Iterable[Any]] = None,
        error: Optional[str] = None,
        stdout: Optional[Any] = None,
        stderr: Optional[Any] = None,
    ) -> None:
        if not self._run:
            return
        summary = {"status": status, "run_id": self.run_id}
        if error:
            summary["error"] = error
        if artifacts:
            summary["artifacts"] = [str(path) for path in artifacts]
        tail_stdout = _tail_output(stdout, lines=200)
        tail_stderr = _tail_output(stderr, lines=200)
        if tail_stdout:
            summary["stdout_tail"] = tail_stdout
        if tail_stderr:
            summary["stderr_tail"] = tail_stderr
        file_logs: Dict[str, str] = {}
        log_payload: Dict[str, Any] = {"status": status}
        if error:
            log_payload["error"] = error
        if tail_stdout is not None:
            log_payload["stdout"] = tail_stdout
        if tail_stderr is not None:
            log_payload["stderr"] = tail_stderr
        stdout_text = self._stringify_output(stdout)
        stderr_text = self._stringify_output(stderr)
        if stdout_text is not None:
            file_logs["stdout.txt"] = stdout_text
        if stderr_text is not None:
            file_logs["stderr.txt"] = stderr_text
        if not self._console_logged:
            self._emit_console_logs(stdout_tail=tail_stdout, stderr_tail=tail_stderr)
        # W&B runs can be auto-finished when multiple threads reinit; avoid crashing if that happens.
        if getattr(self._run, "_is_finished", False):
            return
        try:
            self._run.log(log_payload)
        except Exception:
            pass
        try:
            self._run.summary.update(summary)
        except Exception:
            pass
        # Upload full logs as files so they appear in the Files tab.
        for name, content in file_logs.items():
            path = self._tmp_dir / name
            try:
                path.write_text(content)
                self._run.save(str(path), base_path=str(self._tmp_dir))
            except Exception:
                continue
        try:
            self._run.finish()
        except Exception:
            pass


def run_component_in_container(
    spec: ComponentSpec,
    runner_entry: Optional[Iterable[str]] = None,
    name_prefix: Optional[str] = None,
    runtime: Optional[ContainerRuntimeOptions] = None,
    run_env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    capture_output: bool = False,
    gpus_override: Optional[str] = None,
) -> subprocess.CompletedProcess:
    if spec.container is None:
        raise RuntimeError(f"Component '{spec.name}' missing container configuration")

    runtime = runtime or ContainerRuntimeOptions()
    container = spec.container
    runner = REPO_ROOT / "scripts" / "run_in_container.py"

    cmd: List[str] = [
        sys.executable,
        str(runner),
        "--image",
        container.image,
    ]

    if container.tag:
        cmd.extend(["--tag", container.tag])
    prefix = name_prefix or container.name_prefix
    if prefix:
        cmd.extend(["--name-prefix", prefix])
    gpus_arg = gpus_override if gpus_override is not None else container.gpus
    if gpus_arg:
        cmd.extend(["--gpus", gpus_arg])
    if container.workdir:
        cmd.extend(["--workdir", container.workdir])

    mounts = list(runtime.mounts) + container.mounts
    env_vars = list(runtime.env) + container.env
    if run_env:
        for key, value in run_env.items():
            env_vars.append(f"{key}={value}")
    extra_args = list(runtime.extra_args) + container.extra_args

    has_pythonpath = any(var.startswith("PYTHONPATH=") for var in env_vars)
    if not has_pythonpath:
        workdir = container.workdir or "/workspace"
        default_pythonpath = f"{workdir}/src:{workdir}/profiling/src"
        env_vars.append(f"PYTHONPATH={default_pythonpath}")

    for mount in mounts:
        cmd.extend(["--mount", mount])
    for env_var in env_vars:
        cmd.extend(["--env", env_var])
    for extra in extra_args:
        cmd.append(f"--extra-arg={extra}")

    selected_entry: Iterable[str] = runner_entry or container.runner_entry or runtime.runner_entry
    if isinstance(selected_entry, str):
        selected_entry = (selected_entry,)
    child_args = list(selected_entry)
    cmd.append("--")
    cmd.extend(child_args)

    run_kwargs: Dict[str, Any] = {"check": True}
    if timeout is not None:
        run_kwargs["timeout"] = timeout
    if capture_output:
        run_kwargs["capture_output"] = True
        run_kwargs["text"] = True

    completed = subprocess.run(cmd, **run_kwargs)
    return completed


def run_component(
    spec: ComponentSpec,
    runs: List[Dict[str, Any]],
    runner: ProfilingRunner,
    *,
    smoke_report: Optional[SmokeReport] = None,
    progress: Any = None,
) -> None:
    profiler_cfg = runner.profiler_config() or {}

    requires_nsight = _profiler_requires_nsight(profiler_cfg)
    if requires_nsight and not spec.run_gpu:
        print(
            f"[WARN] Component '{spec.name}' does not expose GPU profiling; skipping runs (profiler requires Nsight)."
        )
        return

    wandb_cfg = runner.wandb_config()
    gpu_module = None

    if not runner.in_container and not runner.dry_run:
        runner.run_runs_in_container(
            spec,
            runs,
            smoke_report=smoke_report,
            progress=progress,
        )
        return

    if not runs:
        print(f"[INFO] No runs selected for component '{spec.name}'.")
        return

    need_progress = not runner.dry_run

    if not runner.dry_run:
        exec_helper = spec.exec_helper or runner.default_exec_helper
        ensure_requirements(
            profiler_cfg,
            need_progress=need_progress,
            exec_helper=exec_helper,
            ignore_nsight=runner.ignore_nsight,
        )
        from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel
    else:
        tqdm = None  # type: ignore

    iterator = tqdm(runs, desc=f"[{spec.name}] runs", unit="run") if (tqdm and progress is None) else runs

    for cfg in iterator:
        start_ts = datetime.utcnow().isoformat()
        meta = _labels_view(cfg)
        detail_parts: List[str] = []
        repeat_idx = meta.get("repeat_idx") or cfg.get("repeat_idx")
        if repeat_idx:
            detail_parts.append(f"repeat={repeat_idx}")
        for key in ("scaffold", "ligand", "quality", "input_batch_size", "output_samples", "total_samples"):
            value = meta.get(key)
            if value:
                detail_parts.append(f"{key}={value}")
        desc = ", ".join(detail_parts)
        header = f"[{spec.name}] run #{cfg.get('run_idx')}: {desc or 'unlabelled'}"
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))

        run_id = runner._format_run_id(spec, cfg)
        cfg["run_id"] = run_id
        scoped_output_root = _scope_output_paths(cfg, spec.name, run_id)
        run_ctx = runner._base_runner_context(run_id)
        if spec.container:
            run_ctx["container"] = runner._as_plain_payload(spec.container)
            cfg["container"] = run_ctx["container"]
        if runner.container_runtime:
            run_ctx["container_runtime"] = runner._as_plain_payload(runner.container_runtime)
            cfg["container_runtime"] = run_ctx["container_runtime"]
        if scoped_output_root is not None:
            run_ctx.setdefault("run_config", {})
            run_ctx["run_config"]["scoped_output_root"] = str(scoped_output_root)
        resolved_artifacts = _shared.resolve_profiler_artifacts(cfg, profiler_cfg, run_ctx)
        cfg["profiler_artifacts"] = [str(path) for path in resolved_artifacts]
        if not runner.skip_existing and not runner.dry_run:
            _reset_run_dirs(cfg)
        runner._write_run_snapshot(cfg, run_ctx, profiler_cfg)

        profiler_artifacts = resolved_artifacts
        profiler_status = _artifact_status(profiler_artifacts)
        marker_ready = _marker_ready(cfg)

        run_profiler_now = not (runner.skip_existing and (profiler_status["ready"] or marker_ready))
        tracker = None
        if not runner.dry_run and run_profiler_now:
            tracker = _start_wandb_run(wandb_cfg, spec.name, cfg, profiler_cfg, run_ctx)

        if runner.dry_run:
            _print_cfg(cfg, profiler_cfg)
            _print_profiler_preview(runner.skip_existing, profiler_status)
            continue

        try:
            if run_profiler_now:
                if gpu_module is None:
                    gpu_module = spec.load_gpu_module()
                _shared.capture_fingerprint_snapshot(cfg)
                _shared.run_gpu_profile(
                    gpu_module,
                    cfg,
                    profiler_cfg=profiler_cfg,
                    runner_context=run_ctx,
                )
            elif runner.skip_existing:
                _print_profiler_skip(profiler_status["present"])
        except Exception as exc:
            if tracker:
                tracker.finish(status="failed", error=str(exc))
            end_ts = datetime.utcnow().isoformat()
            should_continue = smoke_report is not None or runner.continue_on_error
            if isinstance(exc, _shared.MissingNsightTraceError):
                status = "error"
                if smoke_report is not None:
                    runner._record_run_result(
                        component=spec.name,
                        run_id=run_id,
                        status=status,
                        start_time=start_ts,
                        end_time=end_ts,
                        meta=meta,
                        profiler=_profiler_name(profiler_cfg),
                        exit_code=None,
                        error_summary=str(exc),
                        smoke_report=smoke_report,
                    )
                    continue
                if should_continue:
                    continue
            if should_continue:
                runner._record_run_result(
                    component=spec.name,
                    run_id=run_id,
                    status="error",
                    start_time=start_ts,
                    end_time=end_ts,
                    meta=meta,
                    profiler=_profiler_name(profiler_cfg),
                    exit_code=None,
                    error_summary=str(exc),
                    smoke_report=smoke_report,
                )
                continue
            raise
        else:
            if tracker:
                status = "skipped" if (runner.skip_existing and not run_profiler_now) else "completed"
                tracker.finish(status=status, artifacts=profiler_artifacts)
                tracker = None
            if smoke_report is not None:
                end_ts = datetime.utcnow().isoformat()
                runner._record_run_result(
                    component=spec.name,
                    run_id=run_id,
                    status="success",
                    start_time=start_ts,
                    end_time=end_ts,
                    meta=meta,
                    profiler=_profiler_name(profiler_cfg),
                    exit_code=0,
                    error_summary=None,
                    smoke_report=smoke_report,
                )
            if progress:
                progress.update(1)

    if tqdm and progress is None and hasattr(iterator, "close"):
        iterator.close()


def _print_cfg(cfg: Dict[str, Any], profiler_cfg: Any) -> None:
    print("CMD:", " ".join(str(x) for x in cfg.get("CMD", [])))
    print("PARAMS:", " ".join(str(x) for x in cfg.get("PARAMS", [])))
    print("PROF_DIR:", cfg.get("PROF_DIR"))
    print("PROF_DATA_DIR:", cfg.get("PROF_DATA_DIR"))
    print("BASENAME:", cfg.get("BASENAME"))
    if cfg.get("timestamp"):
        print("TIMESTAMP:", cfg["timestamp"])
    if cfg.get("repeat_idx"):
        print("REPEAT_IDX:", cfg["repeat_idx"])
    if cfg.get("run_label"):
        print("RUN_LABEL:", cfg["run_label"])
    if cfg.get("util_csv"):
        print("UTIL_CSV:", cfg["util_csv"])
    profiler_name = _profiler_name(profiler_cfg) or "(default)"
    print("PROFILER:", profiler_name)
    artifacts = cfg.get("profiler_artifacts") or _profiler_artifacts(cfg)
    if artifacts:
        print("ARTIFACTS:", _format_paths(artifacts))


def _print_profiler_preview(skip_existing: bool, status: Dict[str, Any]) -> None:
    if skip_existing:
        if status["ready"]:
            print(f"[profiler] would skip (artifacts present: {_format_paths(status['present'])})")
        else:
            print(f"[profiler] would run (missing: {_format_paths(status['missing'])})")
    else:
        print("[profiler] would run (skip-existing not requested).")


def _print_profiler_skip(artifacts: List[Path]) -> None:
    print("[SKIP] profiler artifacts already exist:")
    for path in artifacts:
        print(f"  - {path}")
    print("       (set runner.skip_existing=false to force a rerun)")


def run_from_config(cfg: DictConfig) -> int:
    runner = instantiate(cfg.runner)
    if hasattr(runner, "bind_config"):
        runner.bind_config(cfg)
    return runner()


@hydra_main(config_path="../../../configs", config_name="run_sweeps", version_base=None)
def hydra_entry(cfg: DictConfig) -> int:
    return run_from_config(cfg)


def main() -> int:
    return hydra_entry()


if __name__ == "__main__":
    sys.exit(main())
