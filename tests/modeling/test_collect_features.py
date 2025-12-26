from __future__ import annotations

from pathlib import Path

import yaml

from modeling.collect_features import collect_feature_rows


def _write_minimal_telemetry_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "timestamp,c1,c2,t0,t1,util,t3,t4,mem_used",
                "2025-01-01T00:00:00Z,x,y,0,0,50,0,0,100",
                "2025-01-01T00:00:10Z,x,y,0,0,70,0,0,200",
            ]
        )
        + "\n"
    )


def _make_successful_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "profile_SUCCESS").write_text("")
    _write_minimal_telemetry_csv(run_dir / "test_telemetry.csv")


def test_collect_features_supports_legacy_no_quality_layout(tmp_path, capsys) -> None:
    raw_root = tmp_path / "raw"
    component = raw_root / "proteinmpnn"
    sample_id = "1n0r_3dx1"

    direct_run = (
        component / "level_a" / sample_id / "input_1" / "output_1" / "run_1"
    )
    quality_run = (
        component
        / "level_a"
        / sample_id
        / "medium"
        / "input_1"
        / "output_1"
        / "run_1"
    )

    _make_successful_run(direct_run)
    _make_successful_run(quality_run)

    rows = collect_feature_rows(
        raw_root=raw_root,
        level="level_a",
        sample_meta={sample_id: {"scaffold_length": 10, "ligand_length": 20}},
    )

    assert len(rows) == 2
    run_paths = {row["run_path"] for row in rows}
    assert str(direct_run) in run_paths
    assert str(quality_run) in run_paths

    for row in rows:
        assert row["quality"] == "medium"
        assert row["input_batch_size"] == 1
        assert row["output_samples"] == 1
        assert row["runtime_sec"] == 10.0
        assert row["peak_memory_mib"] == 200.0
        assert row["mean_temporal_util_percent"] == 60.0

    captured = capsys.readouterr()
    assert "No input_* directories under" not in captured.err


def test_collect_features_aliases_component_output_key(tmp_path) -> None:
    raw_root = tmp_path / "raw"
    component = "diffdock"
    sample_id = "1n0r_3dx1"

    run_dir = (
        raw_root
        / component
        / "level_a"
        / sample_id
        / "medium"
        / "input_1"
        / "output_7"
        / "run_1"
    )
    _make_successful_run(run_dir)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run": {
                    "labels": {
                        "batch_size": 1,
                        "samples_per_complex": 7,
                        "steps": 20,
                    }
                }
            },
            sort_keys=False,
        )
    )

    components_dir = tmp_path / "components"
    components_dir.mkdir(parents=True, exist_ok=True)
    (components_dir / "diffdock.yaml").write_text(
        yaml.safe_dump(
            {
                "name": component,
                "batching": {
                    "input": {"key": "batch_size"},
                    "output": {"key": "samples_per_complex"},
                },
            },
            sort_keys=False,
        )
    )

    rows = collect_feature_rows(
        raw_root=raw_root,
        level="level_a",
        sample_meta={},
        components_dir=components_dir,
    )
    assert len(rows) == 1
    row = rows[0]

    assert row["component"] == "diffdock"
    assert row["output_samples"] == 7
    assert "diffdock_samples_per_complex" not in row
    assert row["diffdock_steps"] == 20
