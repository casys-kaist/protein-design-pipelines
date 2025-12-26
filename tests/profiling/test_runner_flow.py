from __future__ import annotations

import json

from profile.cli import run_sweeps as rs
from omegaconf import OmegaConf
from tests.helpers import DummySpec


def test_runner_lists_runs_without_execution(stubbed_components, monkeypatch, capsys):
    called = False

    def fail_run_component(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("run_component should not be invoked when list_runs=True")

    monkeypatch.setattr(rs, "run_component", fail_run_component)

    runner = rs.ProfilingRunner(
        components=[stubbed_components.name],
        list_runs=True,
        repeats=1,
    )
    rc = runner.run()
    assert rc == 0
    captured = capsys.readouterr().out
    assert "matching runs" in captured
    assert called is False


def test_runner_ignores_unknown_components(monkeypatch):
    monkeypatch.setattr(rs, "available_components", lambda: ["known"])
    runner = rs.ProfilingRunner(components=["known", "bogus"], dry_run=True, repeats=1)
    runner._profiler_cfg = OmegaConf.create({})
    monkeypatch.setattr(runner, "_get_component_spec", lambda name: DummySpec(name="known"))
    monkeypatch.setattr(rs, "run_component", lambda spec, runs, runner, **_kw: None)
    rc = runner.run()
    assert rc == 0


def test_runner_invokes_run_component_in_dry_run(stubbed_components, monkeypatch):
    calls = []

    def stub_run_component(spec, runs, runner, **_kw):
        calls.append((spec.name, json.loads(json.dumps(runs)), runner.dry_run))

    monkeypatch.setattr(rs, "run_component", stub_run_component)

    runner = rs.ProfilingRunner(
        components=[stubbed_components.name],
        dry_run=True,
        repeats=1,
    )
    rc = runner.run()
    assert rc == 0
    assert calls and calls[0][0] == stubbed_components.name
    assert calls[0][2] is True
