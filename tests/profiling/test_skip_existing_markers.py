from __future__ import annotations

from profile.cli import run_sweeps


def test_marker_status_does_not_fall_back_to_results_when_raw_dir_present(tmp_path):
    raw_dir = tmp_path / "raw_run"
    prof_dir = tmp_path / "results_run"
    raw_dir.mkdir()
    prof_dir.mkdir()

    # Simulate a previous success marker in the shared results directory (e.g. level_a),
    # while the profiler-scoped raw directory (e.g. level_c) has not been executed yet.
    (prof_dir / "profile_SUCCESS").write_text("1")

    cfg = {"PROF_DATA_DIR": str(raw_dir), "PROF_DIR": str(prof_dir)}
    assert run_sweeps._marker_status(cfg) is None


def test_marker_status_returns_none_when_raw_dir_missing(tmp_path):
    prof_dir = tmp_path / "results_run"
    prof_dir.mkdir()
    (prof_dir / "profile_SUCCESS").write_text("1")

    cfg = {"PROF_DIR": str(prof_dir)}
    assert run_sweeps._marker_status(cfg) is None
