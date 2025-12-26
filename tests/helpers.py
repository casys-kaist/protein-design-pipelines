from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from profile.components.common import ContainerConfig


@dataclass
class DummySpec:
    """Minimal component spec for runner-level unit tests."""

    name: str = "dummy"
    container: Optional[ContainerConfig] = None
    run_gpu: bool = False

    def build_runs(self, *, timestamp: str, repeats: int) -> list[Dict[str, Any]]:
        labels = {"scaffold": "1abc", "ligand": "xyz", "quality": "low", "run_label": "run_1", "repeat_idx": 1}
        return [
            {
                "labels": labels,
                "meta": labels,
                "combo": labels,
                "run_idx": 1,
                "run_label": "run_1",
                "timestamp": timestamp,
                "CMD": ["echo", "hello"],
                "PARAMS": [],
                "PROF_DIR": "dummy",
                "PROF_DATA_DIR": "dummy",
                "BASENAME": "dummy",
                "PREP_COMMANDS": [],
                "SAMPLE_INTERVAL": 0.1,
                "START_INDEX": 1,
                "util_csv": "dummy.csv",
            }
        ]


def sample_run(tmp_path: Path, *, run_idx: int = 1, meta: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Build a realistic run payload for container execution tests."""
    meta = meta or {"scaffold": "1abc", "ligand": "xyz", "quality": "low"}
    labels = dict(meta)
    labels.setdefault("run_label", f"run_{run_idx}")
    labels.setdefault("repeat_idx", run_idx)
    prof_dir = tmp_path / f"prof_{run_idx}"
    data_dir = tmp_path / f"data_{run_idx}"
    return {
        "labels": labels,
        "meta": labels,
        "combo": labels,
        "run_idx": run_idx,
        "run_label": f"run_{run_idx}",
        "timestamp": "20240101_000000",
        "CMD": ["echo", "hello"],
        "PARAMS": [],
        "PROF_DIR": str(prof_dir),
        "PROF_DATA_DIR": str(data_dir),
        "BASENAME": "dummy",
        "PREP_COMMANDS": [],
        "SAMPLE_INTERVAL": 0.1,
        "START_INDEX": 1,
        "util_csv": str(prof_dir / "dummy_utilization.csv"),
    }
