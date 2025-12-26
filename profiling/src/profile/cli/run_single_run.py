#!/usr/bin/env python3
"""Execute a single profiling run inside a container (payload-first, Hydra fallback)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

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

from profile.cli import run_sweeps as sweeps  # noqa: E402
from profile.components import _shared, load_component_spec  # noqa: E402


def _describe_meta(meta: Dict[str, Any]) -> str:
    detail_parts: List[str] = []
    repeat_idx = meta.get("repeat_idx") or meta.get("repeat")
    if repeat_idx:
        detail_parts.append(f"repeat={repeat_idx}")
    for key in ("scaffold", "ligand", "quality", "input_batch_size", "output_samples", "total_samples", "run_label"):
        value = meta.get(key)
        if value:
            detail_parts.append(f"{key}={value}")
    return ", ".join(detail_parts)


def _select_run(run_cfgs: List[Dict[str, Any]], run_filters: Dict[str, Any]) -> Dict[str, Any]:
    target_repeat = run_filters.get("repeat_idx") or run_filters.get("repeat")
    if target_repeat is not None:
        try:
            target_repeat = int(target_repeat)
        except ValueError:
            target_repeat = None
    if target_repeat is not None:
        filtered = [
            cfg
            for cfg in run_cfgs
            if (cfg.get("repeat_idx") or cfg.get("meta", {}).get("repeat_idx")) == target_repeat
        ]
        if filtered:
            return filtered[0]
    return run_cfgs[0]


def _ensure_component(cfg: DictConfig) -> str:
    component = cfg.get("component")
    if component:
        return str(component)
    runner_node = cfg.get("runner") or {}
    components = runner_node.get("components") or []
    if components:
        return str(components[0])
    raise ValueError("Component name is required (pass `component=<name>` override).")


def _run_from_payload(payload_path: Path) -> int:
    payload = json.loads(Path(payload_path).read_text())
    component = payload.get("component")
    if not component:
        raise ValueError("payload is missing component")

    spec = load_component_spec(component)
    run_cfg = payload.get("run_cfg") or {}
    run_id = payload.get("run_id") or run_cfg.get("run_id") or run_cfg.get("BASENAME") or component
    run_cfg["run_id"] = run_id
    runner_ctx = payload.get("runner_ctx") or {}
    profiler_cfg = payload.get("profiler_cfg")
    skip_existing = bool(payload.get("skip_existing"))
    if profiler_cfg is not None and not isinstance(profiler_cfg, DictConfig):
        profiler_cfg = OmegaConf.create(profiler_cfg)
    wandb_cfg = payload.get("wandb") if payload.get("enable_wandb_in_container") else None
    if wandb_cfg is not None and not isinstance(wandb_cfg, sweeps.WandbConfig):
        wandb_cfg = sweeps.WandbConfig(**wandb_cfg)

    resolved_artifacts = _shared.resolve_profiler_artifacts(run_cfg, profiler_cfg, runner_ctx)
    run_cfg["profiler_artifacts"] = [str(path) for path in resolved_artifacts]
    tracker = None

    exec_helper = payload.get("exec_helper")
    ignore_nsight = bool(payload.get("ignore_nsight", False))
    sweeps.ensure_requirements(
        profiler_cfg,
        need_progress=False,
        exec_helper=exec_helper,
        ignore_nsight=ignore_nsight,
    )

    if skip_existing:
        status = sweeps._artifact_status(resolved_artifacts)
        if sweeps._should_skip_existing(resolved_artifacts, run_cfg):
            sweeps._print_profiler_skip(status["present"] or resolved_artifacts)
            return 0
    else:
        sweeps._reset_run_dirs(run_cfg)

    tracker = sweeps._start_wandb_run(
        wandb_cfg,
        component,
        run_cfg,
        profiler_cfg,
        runner_ctx,
    )
    gpu_module = spec.load_gpu_module()
    try:
        _shared.capture_fingerprint_snapshot(run_cfg)
        _shared.run_gpu_profile(
            gpu_module,
            run_cfg,
            profiler_cfg=profiler_cfg,
            runner_context=runner_ctx,
        )
    except _shared.MissingNsightTraceError as exc:
        if tracker:
            tracker.finish(status="failed", error=str(exc))
        print(f"[ERROR] {exc}")
        raise SystemExit(1)
    except subprocess.TimeoutExpired as exc:
        if tracker:
            tracker.finish(status="timeout", error=str(exc))
        print(f"[ERROR] {exc}")
        raise SystemExit(124)
    except Exception as exc:
        if tracker:
            tracker.finish(status="failed", error=str(exc))
        raise
    else:
        if tracker:
            tracker.finish(status="completed", artifacts=resolved_artifacts)
    return 0


def _run_single(cfg: DictConfig) -> int:
    runner = instantiate(cfg.runner)
    if hasattr(runner, "bind_config"):
        runner.bind_config(cfg)
    runner.in_container = True

    component = _ensure_component(cfg)
    spec = load_component_spec(component)
    profiler_cfg = runner.profiler_config()
    skip_existing = bool(runner.skip_existing)
    if profiler_cfg is None:
        raise RuntimeError("Profiler config is required; set `profiler=level_a|level_b|level_c`.")

    requires_nsight = sweeps._profiler_requires_nsight(profiler_cfg)
    if requires_nsight and not spec.run_gpu:
        print(f"[WARN] Component '{component}' does not expose GPU profiling; skipping (profiler requires Nsight).")
        return 0

    run_overrides = OmegaConf.to_container(cfg.get("run") or {}, resolve=True) or {}
    combos = [run_overrides] if run_overrides else None
    timestamp = getattr(cfg.runner, "timestamp", None)
    repeats = getattr(cfg.runner, "repeats", 1)
    profiler_label = sweeps._profiler_name(profiler_cfg)
    run_cfgs = spec.build_runs(
        timestamp=timestamp,
        repeats=repeats,
        combos=combos,
        profiler_label=profiler_label,
    )
    if not run_cfgs:
        raise RuntimeError(f"No runs generated for component '{component}' (check run overrides).")
    if combos is None and len(run_cfgs) > 1:
        raise RuntimeError("Expected a single run; provide run.* overrides to select one combination.")
    run_cfg = _select_run(run_cfgs, run_overrides)

    run_id = cfg.get("run_id")
    if not run_id:
        run_id = runner._format_run_id(spec, run_cfg)
    run_cfg["run_id"] = run_id
    runner_ctx = runner._base_runner_context(run_id)

    resolved_artifacts = _shared.resolve_profiler_artifacts(run_cfg, profiler_cfg, runner_ctx)
    run_cfg["profiler_artifacts"] = [str(path) for path in resolved_artifacts]

    header = f"[{component}] run_id={run_id}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    meta_desc = _describe_meta(run_cfg.get("meta", {}))
    if meta_desc:
        print(f"META: {meta_desc}")

    profiler_status = sweeps._artifact_status(resolved_artifacts)
    if skip_existing and sweeps._should_skip_existing(resolved_artifacts, run_cfg):
        sweeps._print_profiler_skip(profiler_status["present"] or resolved_artifacts)
        print("[SKIP] Requested artifacts already exist for this run.")
        return 0

    if not skip_existing and not runner.dry_run:
        sweeps._reset_run_dirs(run_cfg)

    exec_helper = spec.exec_helper or runner.default_exec_helper
    sweeps.ensure_requirements(
        profiler_cfg,
        need_progress=False,
        exec_helper=exec_helper,
        ignore_nsight=runner.ignore_nsight,
    )

    gpu_module = spec.load_gpu_module()

    try:
        _shared.capture_fingerprint_snapshot(run_cfg)
        _shared.run_gpu_profile(
            gpu_module,
            run_cfg,
            profiler_cfg=profiler_cfg,
            runner_context=runner_ctx,
        )
    except _shared.MissingNsightTraceError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)
    except subprocess.TimeoutExpired as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(124)
    except Exception as exc:
        raise
    return 0


@hydra_main(config_path="../../../configs", config_name="run_sweeps", version_base=None)
def hydra_entry(cfg: DictConfig) -> int:
    return _run_single(cfg)


def main() -> int:
    argv = sys.argv[1:]
    if "--payload" in argv:
        idx = argv.index("--payload")
        try:
            payload_path = argv[idx + 1]
        except IndexError as exc:  # pragma: no cover - defensive
            raise SystemExit("Missing payload path for --payload") from exc
        return _run_from_payload(Path(payload_path))
    return hydra_entry()


if __name__ == "__main__":
    raise SystemExit(main())
