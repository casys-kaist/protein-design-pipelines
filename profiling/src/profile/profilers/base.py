"""Base interfaces for profiler strategies used by the sweep runner."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ProfilerStrategy(ABC):
    """Base interface for multi-level profiling strategies."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        merged: Dict[str, Any] = dict(config or {})
        merged.update(kwargs)
        self.config: Dict[str, Any] = merged

    @abstractmethod
    def build_command(self, original_cmd: List[str], output_path: str) -> List[str]:
        """Return the wrapped command that should be executed."""

    @abstractmethod
    def pre_run_hook(self, meta: Dict[str, Any]) -> None:
        """Called before executing the profiling command."""

    @abstractmethod
    def post_run_hook(self, meta: Dict[str, Any]) -> None:
        """Called after the command finishes (successfully or otherwise)."""


__all__ = ["ProfilerStrategy"]
