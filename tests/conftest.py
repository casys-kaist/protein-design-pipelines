from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the profiling package is importable without an editable install.
TESTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TESTS_ROOT.parents[0]
PROFILING_SRC = REPO_ROOT / "profiling" / "src"
if str(PROFILING_SRC) not in sys.path:
    sys.path.insert(0, str(PROFILING_SRC))

from profile.cli import run_sweeps as rs  # noqa: E402

from .helpers import DummySpec


@pytest.fixture
def dummy_spec() -> DummySpec:
    return DummySpec()


@pytest.fixture
def stubbed_components(monkeypatch, dummy_spec: DummySpec) -> DummySpec:
    """Provide a stub spec + predictable available-components output."""
    monkeypatch.setattr(rs, "available_components", lambda: [dummy_spec.name])
    monkeypatch.setattr(rs, "load_component_spec", lambda name: dummy_spec)
    return dummy_spec
