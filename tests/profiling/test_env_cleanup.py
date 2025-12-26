from profile.components import _shared


def test_prepare_run_environment_strips_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/tmp/custom")
    env, mps = _shared._prepare_run_environment({"run_config": {}}, {"PROF_DIR": "/tmp", "PROF_DATA_DIR": "/tmp", "BASENAME": "run"})
    assert "PYTHONPATH" not in env
    assert mps is None


def test_prepare_run_environment_parses_stringified_dict_env():
    env_str = "{'FOO': 'bar', 'CUDA_VISIBLE_DEVICES': '2'}"
    env, mps = _shared._prepare_run_environment({"run_config": {"env": env_str}}, {"PROF_DIR": "/tmp", "PROF_DATA_DIR": "/tmp", "BASENAME": "run"})
    assert env["FOO"] == "bar"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert mps is None
