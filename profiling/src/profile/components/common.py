from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from profile import TOOLS_ROOT

from . import _shared
from .batching import BatchDimension, BatchingConfig

ApplyRulesFn = Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]


@dataclass
class ContainerConfig:
    image: str
    tag: str = "profiling"
    mounts: List[str] = field(default_factory=list)
    env: List[str] = field(default_factory=list)
    extra_args: List[str] = field(default_factory=list)
    gpus: Optional[str] = "all"
    workdir: str = "/workspace"
    name_prefix: str = "profile-sweep"
    runner_entry: Optional[List[str]] = None


@dataclass
class ComponentSpec:
    """Declarative definition for a profiling component."""

    name: str
    templates: Dict[str, Any]
    apply_rules: Optional[ApplyRulesFn] = None
    run_gpu: bool = True
    max_workers: Optional[int] = None
    exec_helper: Optional[str] = None
    container: Optional[ContainerConfig] = None
    batching: Optional[BatchingConfig] = None
    sys_util_path: str = field(default_factory=lambda: str(TOOLS_ROOT / "system_utilization.py"))
    gpu_profile_path: Optional[str] = field(default_factory=lambda: str(TOOLS_ROOT / "gpu_profile.py"))

    def build_runs(
        self,
        *,
        timestamp: Optional[str] = None,
        repeats: int = 1,
        combos: Optional[List[Dict[str, Any]]] = None,
        profiler_label: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Expand templates into concrete run dictionaries."""
        normalized_combos = _shared.normalize_combos(combos)
        if not normalized_combos:
            raise ValueError(
                f"No run combinations provided for component '{self.name}'. "
                "Define experiments.<component> entries or pass run.* overrides."
            )
        return _shared.build_runs(
            self.templates,
            normalized_combos,
            self.apply_rules,
            timestamp=timestamp,
            repeats=repeats,
            batching=self.batching,
            profiler_label=profiler_label,
        )

    def load_system_module(self):
        if not self.sys_util_path:
            return None
        return _shared.load_module_from_path(self.sys_util_path, f"sys_util_{self.name}")

    def load_gpu_module(self):
        if not self.run_gpu or not self.gpu_profile_path:
            return None
        return _shared.load_module_from_path(self.gpu_profile_path, f"gpu_profile_{self.name}")
