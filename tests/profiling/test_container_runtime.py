from __future__ import annotations

import pytest

from profile.cli import run_sweeps as rs
from profile.components.common import ContainerConfig

from tests.helpers import DummySpec, sample_run


def _capture_container_command(monkeypatch, runner_entry):
    recorded = {}

    def fake_run(cmd, check):  # noqa: D401 - helper stub
        recorded["cmd"] = cmd

    monkeypatch.setattr(rs.subprocess, "run", fake_run)

    spec = DummySpec(container=ContainerConfig(image="dummy", runner_entry=runner_entry))

    runtime = rs.ContainerRuntimeOptions(mounts=[], env=[], extra_args=[])
    rs.run_component_in_container(spec, runner_entry=None, runtime=runtime)
    return recorded["cmd"]


def test_run_component_in_container_defaults_to_python3(monkeypatch):
    cmd = _capture_container_command(monkeypatch, runner_entry=None)
    idx = cmd.index("--")
    assert cmd[idx + 1 :] == ["python3", "-m", "profile.cli.run_single_run"]


def test_run_component_in_container_honors_runner_entry(monkeypatch):
    custom_entry = ["python3.9", "-m", "profile.cli.run_sweeps"]
    cmd = _capture_container_command(monkeypatch, runner_entry=custom_entry)
    idx = cmd.index("--")
    assert cmd[idx + 1 :] == custom_entry


def test_pythonpath_injected_when_missing(monkeypatch):
    recorded = {}

    def fake_run(cmd, check):
        recorded["cmd"] = cmd

    monkeypatch.setattr(rs.subprocess, "run", fake_run)

    spec = DummySpec(container=ContainerConfig(image="dummy", workdir="/workspace"))
    runtime = rs.ContainerRuntimeOptions(mounts=[], env=[], extra_args=[])
    rs.run_component_in_container(spec, runner_entry=None, runtime=runtime)

    assert any(
        "PYTHONPATH=/workspace/src:/workspace/profiling/src" in arg for arg in recorded["cmd"]
    )


def test_run_component_includes_run_env(monkeypatch):
    recorded = {}

    def fake_run(cmd, check):
        recorded["cmd"] = cmd

    monkeypatch.setattr(rs.subprocess, "run", fake_run)

    spec = DummySpec(container=ContainerConfig(image="dummy", workdir="/workspace"))
    runtime = rs.ContainerRuntimeOptions(mounts=[], env=[], extra_args=[])
    rs.run_component_in_container(
        spec,
        runner_entry=None,
        runtime=runtime,
        run_env={"TRITON_CACHE_DIR": "/cache/triton"},
    )

    env_vars = [recorded["cmd"][idx + 1] for idx, token in enumerate(recorded["cmd"]) if token == "--env"]
    assert "TRITON_CACHE_DIR=/cache/triton" in env_vars


def test_run_component_allows_gpu_override(monkeypatch):
    recorded = {}

    def fake_run(cmd, check):
        recorded["cmd"] = cmd

    monkeypatch.setattr(rs.subprocess, "run", fake_run)

    spec = DummySpec(container=ContainerConfig(image="dummy", gpus="all"))
    runtime = rs.ContainerRuntimeOptions(mounts=[], env=[], extra_args=[])
    rs.run_component_in_container(
        spec,
        runner_entry=None,
        runtime=runtime,
        gpus_override="device=2",
    )

    idx = recorded["cmd"].index("--gpus")
    assert recorded["cmd"][idx + 1] == "device=2"


def test_run_component_requires_container_for_real_runs():
    runner = rs.ProfilingRunner(components=["dummy"])
    runner.in_container = False
    with pytest.raises(RuntimeError, match="must define `container`"):
        rs.run_component(DummySpec(), [{"meta": {}, "run_idx": 1}], runner)


def test_run_component_dry_run_allowed_without_container(tmp_path):
    runner = rs.ProfilingRunner(components=["dummy"], dry_run=True)
    runner.in_container = False
    rs.run_component(DummySpec(), [sample_run(tmp_path)], runner)


def test_run_runs_in_container_launches_per_run(monkeypatch, tmp_path):
    recorded = []

    def fake_persist(self, spec, run_cfg, run_id, *_args, **_kwargs):
        payload_path = tmp_path / f"{run_id}.json"
        payload_path.write_text("{}")
        return payload_path, f"/workspace/{payload_path.name}"

    def fake_run_container(spec, **kwargs):
        recorded.append((spec.name, kwargs))

    monkeypatch.setattr(rs.ProfilingRunner, "_persist_run_payload", fake_persist)
    monkeypatch.setattr(rs, "run_component_in_container", fake_run_container)

    spec = DummySpec(container=ContainerConfig(image="dummy", name_prefix="pref"))
    runner = rs.ProfilingRunner(components=["dummy"])

    run_a = sample_run(tmp_path, run_idx=1)
    run_b = sample_run(
        tmp_path,
        run_idx=2,
        meta={"scaffold": "1tim", "ligand": "abc", "quality": "high"},
    )

    runner.run_runs_in_container(spec, [run_a, run_b], True, False)

    assert len(recorded) == 2
    for name, kwargs in recorded:
        assert name == "dummy"
        runner_entry = kwargs["runner_entry"]
        assert runner_entry[:3] == ("python3", "-m", "profile.cli.run_single_run")
        assert runner_entry[3] == "--payload"
        assert runner_entry[-1].startswith("/workspace/")
        assert kwargs["name_prefix"].startswith("pref-")
