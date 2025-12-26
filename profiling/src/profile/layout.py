from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROFILE_ROOT = Path(__file__).resolve().parent
COMPONENTS_ROOT = PROFILE_ROOT / "components"
TOOLS_ROOT = PROFILE_ROOT / "tools"
CATALOGS_ROOT = PROFILE_ROOT / "catalogs"
DOCS_ROOT = PROFILE_ROOT / "docs"

DEFAULT_STORAGE_ROOT = Path("/mnt/nfs/new/bioinformatics/profile")
RESULTS_SUBDIR = "results"
RAW_SUBDIR = "raw"
TRACES_SUBDIR = RAW_SUBDIR  # Backwards-compat alias for legacy code/paths.
SCRATCH_SUBDIR = "scratch"
INPUTS_SUBDIR = "inputs"
DEFAULT_RESULTS_ROOT = DEFAULT_STORAGE_ROOT / RESULTS_SUBDIR
DEFAULT_RAW_ROOT = DEFAULT_STORAGE_ROOT / RAW_SUBDIR
DEFAULT_TRACES_ROOT = DEFAULT_RAW_ROOT
DEFAULT_OUTPUT_ROOT = DEFAULT_STORAGE_ROOT / SCRATCH_SUBDIR
DEFAULT_INPUT_ROOT = DEFAULT_STORAGE_ROOT / INPUTS_SUBDIR
DEFAULT_DATASET_ROOT = DEFAULT_STORAGE_ROOT.parent

# Backwards-compatible default aliases.
DEFAULT_DATA_ROOT = DEFAULT_RESULTS_ROOT


@dataclass(frozen=True)
class ProfileRoots:
    """Resolved directories used by the profiling utilities."""

    storage: Path
    results: Path
    traces: Path
    scratch: Path
    inputs: Path
    datasets: Path

    @classmethod
    def from_environment(cls) -> "ProfileRoots":
        storage_env = os.environ.get("PROFILE_STORAGE_ROOT")
        storage = Path(storage_env).expanduser() if storage_env else DEFAULT_STORAGE_ROOT

        results = storage / RESULTS_SUBDIR
        traces = storage / TRACES_SUBDIR
        scratch = storage / SCRATCH_SUBDIR
        inputs = storage / INPUTS_SUBDIR
        datasets = storage.parent if storage.parent != storage else storage

        return cls(
            storage=storage,
            results=results,
            traces=traces,
            scratch=scratch,
            inputs=inputs,
            datasets=datasets,
        )


ROOTS = ProfileRoots.from_environment()
STORAGE_ROOT = ROOTS.storage
RESULTS_ROOT = ROOTS.results
TRACES_ROOT = ROOTS.traces
SCRATCH_ROOT = ROOTS.scratch
INPUT_ROOT = ROOTS.inputs
DATASET_ROOT = ROOTS.datasets

# Backwards-compatible aliases used throughout the repo.
DATA_ROOT = RESULTS_ROOT
RAW_DATA_ROOT = TRACES_ROOT
OUTPUT_ROOT = SCRATCH_ROOT


def component_output_root(component: str) -> Path:
    """Return the directory under DATA_ROOT for processed outputs (legacy)."""
    return DATA_ROOT / component


def component_raw_root(component: str) -> Path:
    """Return the directory under RAW_DATA_ROOT that should contain raw profiler artifacts."""
    return RAW_DATA_ROOT / component

__all__ = [
    "CATALOGS_ROOT",
    "COMPONENTS_ROOT",
    "DATASET_ROOT",
    "DATA_ROOT",
    "DOCS_ROOT",
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_INPUT_ROOT",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_RAW_ROOT",
    "DEFAULT_RESULTS_ROOT",
    "DEFAULT_STORAGE_ROOT",
    "DEFAULT_TRACES_ROOT",
    "INPUT_ROOT",
    "OUTPUT_ROOT",
    "PROFILE_ROOT",
    "ProfileRoots",
    "RESULTS_ROOT",
    "ROOTS",
    "RAW_DATA_ROOT",
    "SCRATCH_ROOT",
    "STORAGE_ROOT",
    "TOOLS_ROOT",
    "TRACES_ROOT",
    "component_output_root",
    "component_raw_root",
]
