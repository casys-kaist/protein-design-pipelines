from types import SimpleNamespace
from pathlib import Path

import pytest

from profile.cli.run_sweeps import ProfilingRunner


def test_build_cli_overrides_uses_profiler_override(monkeypatch):
    runner = ProfilingRunner(repeats=2, mode="smoke", repo_root=Path("/tmp"))
    runner._profiler_override = "profiler=level_a"

    cfg = {
        "labels": {"sample_id": "sampleX", "quality": "medium", "repeat_idx": 1},
        "repeat_idx": 1,
    }
    spec = SimpleNamespace(name="rfdiffusion")
    run_id = "rfdiffusion-sampleX-medium-r1"

    overrides = runner._build_cli_overrides(spec, cfg, run_id)

    assert "profiler=level_a" in overrides
    assert "+run.sample_id=sampleX" in overrides
    assert "+run.quality=medium" in overrides
