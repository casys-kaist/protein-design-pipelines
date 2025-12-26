from __future__ import annotations

from profile.components import _shared


def test_run_prep_commands_rewrites_python_to_sys_executable(monkeypatch):
    calls = []

    def fake_run(cmd, *, shell, check):
        calls.append((cmd, shell, check))

    monkeypatch.setattr(_shared.subprocess, "run", fake_run)
    monkeypatch.setattr(_shared.sys, "executable", "/opt/conda/bin/python")

    _shared._run_prep_commands(["python /tmp/script.py --flag 1"])

    assert calls == [("/opt/conda/bin/python /tmp/script.py --flag 1", True, True)]


def test_run_prep_commands_leaves_non_python_commands_unchanged(monkeypatch):
    calls = []

    def fake_run(cmd, *, shell, check):
        calls.append((cmd, shell, check))

    monkeypatch.setattr(_shared.subprocess, "run", fake_run)

    _shared._run_prep_commands(["echo hello"])

    assert calls == [("echo hello", True, True)]
