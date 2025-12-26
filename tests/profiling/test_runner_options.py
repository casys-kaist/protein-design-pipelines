from __future__ import annotations

import types
import subprocess
import pytest
from unittest.mock import ANY

from pathlib import Path

from omegaconf import OmegaConf

from profile.cli import run_sweeps as rs
from profile.components import _shared
from profile.components.batching import BatchDimension, BatchingConfig
from profile.components.common import ContainerConfig
from profile.profilers import strategies
from tests.helpers import DummySpec, sample_run

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_quality_level_injected_into_runs(tmp_path):
    templates = {
        "CMD": ["echo", "hello"],
        "PARAMS": [],
        "PROF_DIR": str(tmp_path / "prof"),
        "PROF_DATA_DIR": str(tmp_path / "data"),
        "BASENAME": "test",
    }
    combos = {"quality": ["low", "medium", "high"]}

    runs = _shared.build_runs(templates, combos, None, timestamp="20240101", repeats=1)

    assert [run["labels"].get("quality_level") for run in runs] == [1, 2, 3]
    assert all(run["meta"] is run["labels"] for run in runs)
    assert all(run["combo"] is run["labels"] for run in runs)
    # Ensure the original quality labels are preserved alongside numeric levels.
    assert [run["labels"].get("quality") for run in runs] == ["low", "medium", "high"]


def test_quality_defaults_to_medium_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(_shared, "RAW_DATA_ROOT", tmp_path / "raw")

    templates = {
        "CMD": ["echo", "hello"],
        "PARAMS": [],
        "PROF_DIR": str(tmp_path / "raw" / "proteinmpnn" / "{sample_id}" / "{quality}"),
        "PROF_DATA_DIR": str(tmp_path / "raw" / "proteinmpnn" / "{sample_id}" / "{quality}"),
        "BASENAME": "proteinmpnn_{quality}",
    }

    runs = _shared.build_runs(
        templates,
        {"sample_id": ["1abc_2def"]},
        None,
        timestamp="20240101",
        repeats=1,
        profiler_label="level_a",
    )
    assert runs, "expected run generated when quality is omitted"
    run = runs[0]
    labels = run["labels"]

    assert labels.get("quality") == "medium"
    assert labels.get("quality_level") == 2
    assert "medium" in Path(run["PROF_DIR"]).parts
    assert "medium" in Path(run["PROF_DATA_DIR"]).parts


def test_batch_metadata_defaults_and_product(tmp_path):
    templates = {
        "CMD": ["echo", "{input_bs}"],
        "PARAMS": ["--samples", "{num_samples}"],
        "PROF_DIR": str(tmp_path / "prof_{input_bs}"),
        "PROF_DATA_DIR": str(tmp_path / "data"),
        "BASENAME": "test",
    }
    batching = BatchingConfig(
        input=BatchDimension(key="input_bs", default=4),
        output=BatchDimension(key="num_samples", default=2),
    )

    runs = _shared.build_runs(templates, [{}], None, timestamp="20240101", repeats=1, batching=batching)
    assert runs, "expected a single run with defaults applied"
    run = runs[0]

    assert run["labels"].get("input_batch_size") == 4
    assert run["labels"].get("output_samples") == 2
    assert run["labels"].get("total_samples") == 8
    assert run["labels"].get("input_bs") == 4
    assert run["labels"].get("num_samples") == 2


def test_batch_dirs_and_run_id_include_input_output(monkeypatch, tmp_path):
    monkeypatch.setattr(_shared, "RAW_DATA_ROOT", tmp_path / "raw")

    templates = {
        "CMD": ["echo", "hello"],
        "PARAMS": [],
        "PROF_DIR": str(tmp_path / "raw" / "mmseqs2" / "{sample_id}" / "{quality}"),
        "PROF_DATA_DIR": str(tmp_path / "raw" / "mmseqs2" / "{sample_id}" / "{quality}"),
        "BASENAME": "mmseqs2",
    }
    batching = BatchingConfig(
        input=BatchDimension(key="batch_size", default=1),
        output=BatchDimension(key="num_samples", default=1),
    )

    runs = _shared.build_runs(
        templates,
        {"sample_id": ["1shg_4de1"], "quality": ["medium"]},
        None,
        timestamp="20240101",
        repeats=1,
        batching=batching,
        profiler_label="level_a",
    )
    run = runs[0]

    expected_dir = tmp_path / "raw" / "mmseqs2" / "level_a" / "1shg_4de1" / "medium" / "input_1" / "output_1" / "run_1"
    assert Path(run["PROF_DIR"]) == expected_dir
    assert Path(run["PROF_DATA_DIR"]) == expected_dir

    runner = rs.ProfilingRunner(components=["mmseqs2"], config_cache_dir=tmp_path / ".hydra_cache")
    spec = DummySpec(name="mmseqs2")
    meta = {"sample_id": "1shg_4de1", "quality": "medium", "input_batch_size": 1, "output_samples": 1}
    cfg = sample_run(tmp_path, meta=meta, run_idx=1)
    run_id = runner._format_run_id(spec, cfg)
    assert run_id == "mmseqs2-1shg_4de1-medium-inp1-out1-r1"


def test_batch_dirs_include_output_one_for_input_only(monkeypatch, tmp_path):
    monkeypatch.setattr(_shared, "RAW_DATA_ROOT", tmp_path / "raw")

    templates = {
        "CMD": ["echo", "hello"],
        "PARAMS": [],
        "PROF_DIR": str(tmp_path / "raw" / "esmfold" / "{sample_id}" / "{quality}"),
        "PROF_DATA_DIR": str(tmp_path / "raw" / "esmfold" / "{sample_id}" / "{quality}"),
        "BASENAME": "esmfold",
    }
    batching = BatchingConfig(input=BatchDimension(key="batch_size", default=4), output=None)

    runs = _shared.build_runs(
        templates,
        {"sample_id": ["1shg_4de1"], "quality": ["medium"]},
        None,
        timestamp="20240101",
        repeats=1,
        batching=batching,
        profiler_label="level_a",
    )
    run = runs[0]

    expected_dir = tmp_path / "raw" / "esmfold" / "level_a" / "1shg_4de1" / "medium" / "input_4" / "output_1" / "run_1"
    assert Path(run["PROF_DIR"]) == expected_dir
    assert Path(run["PROF_DATA_DIR"]) == expected_dir

    labels = run["labels"]
    assert labels.get("input_batch_size") == 4
    assert labels.get("output_samples") == 1
    assert labels.get("total_samples") == 4


def test_collapse_output_samples_dedupes():
    collapsed = rs._collapse_output_samples(
        [
            {"sample_id": "1shg_4de1", "quality": "medium", "output_samples": 8},
            {"sample_id": "1shg_4de1", "quality": "medium", "output_samples": 16},
        ],
        value=1,
    )
    assert collapsed == [{"sample_id": "1shg_4de1", "quality": "medium", "output_samples": 1}]


def test_marker_ready_checks_legacy_paths(tmp_path):
    # Primary path (with batch) has no marker; legacy path without batch does.
    new_path = tmp_path / "raw" / "rfdiffusion" / "level_a" / "1shg_4de1" / "medium" / "input_1" / "output_1" / "run_1"
    legacy_path = tmp_path / "raw" / "rfdiffusion" / "level_a" / "1shg_4de1" / "medium" / "run_1"
    legacy_path.mkdir(parents=True)
    (legacy_path / "profile_SUCCESS").write_text("success")

    cfg = {"PROF_DATA_DIR": str(new_path)}
    assert rs._marker_ready(cfg) is True


def test_diffdock_batch_prep_script(tmp_path):
    csv_path = tmp_path / "batch.csv"
    protein = tmp_path / "protein.pdb"
    ligand = tmp_path / "ligand.mol2"
    protein.write_text("PROTEIN")
    ligand.write_text("LIGAND")

    cmd = [
        "python",
        str(REPO_ROOT / "profiling" / "src" / "profile" / "components" / "prep_diffdock_batch.py"),
        "--csv-path",
        str(csv_path),
        "--protein-path",
        str(protein),
        "--ligand-description",
        str(ligand),
        "--complex-name",
        "foo",
        "--count",
        "3",
    ]
    subprocess.run(cmd, check=True)

    rows = csv_path.read_text().strip().splitlines()
    assert len(rows) == 4  # header + 3 rows
    assert rows[0].split(",") == ["complex_name", "protein_path", "protein_sequence", "ligand_description"]


def test_mmseqs_batch_prep_script(tmp_path):
    src = tmp_path / "source.fasta"
    dest = tmp_path / "batched.fasta"
    src.write_text(">seq1\nAAACCC\n")

    cmd = [
        "python",
        str(REPO_ROOT / "profiling" / "src" / "profile" / "components" / "prep_mmseqs_batch.py"),
        "--source",
        str(src),
        "--dest",
        str(dest),
        "--count",
        "3",
    ]
    subprocess.run(cmd, check=True)

    lines = dest.read_text().strip().splitlines()
    headers = [line for line in lines if line.startswith(">")]
    seqs = [line for line in lines if not line.startswith(">")]
    assert len(headers) == 3
    assert len(set(headers)) == 3, "batched FASTA should deduplicate headers"
    assert seqs == ["AAACCC"] * 3


def test_wandb_tracker_records_status(monkeypatch):
    created_runs = []

    class DummyRun:
        def __init__(self, kwargs):
            self.kwargs = kwargs
            self.summary = {}
            self.logged = []

        def log(self, payload):
            self.logged.append(payload)

        def finish(self):
            self.logged.append({"finished": True})

    def fake_init(**kwargs):
        dummy = DummyRun(kwargs)
        created_runs.append(dummy)
        return dummy

    fake_module = types.SimpleNamespace(init=fake_init)
    monkeypatch.setitem(rs.sys.modules, "wandb", fake_module)

    cfg = rs.WandbConfig(enabled=True, project="proj", entity="ent", tags=["tag"], mode="offline")
    run_cfg = {"run_id": "run123", "labels": {"quality": "high", "quality_level": 3}}
    tracker = rs._start_wandb_run(cfg, "comp", run_cfg, profiler_cfg=None, runner_ctx={"run_config": {}})

    assert tracker is not None
    tracker.finish(status="completed", artifacts=["artifact.csv"])

    assert created_runs, "wandb.init was not called"
    recorded = created_runs[0].kwargs
    assert "quality_level:3" in recorded.get("tags", [])
    assert recorded.get("name") == "run123"
    assert created_runs[0].summary.get("status") == "completed"


def test_wandb_shadow_module_does_not_crash(monkeypatch):
    monkeypatch.setitem(rs.sys.modules, "wandb", types.SimpleNamespace())

    cfg = rs.WandbConfig(enabled=True, project="proj", entity="ent", tags=["tag"], mode="offline")
    run_cfg = {"run_id": "run123", "labels": {}}
    tracker = rs._start_wandb_run(cfg, "comp", run_cfg, profiler_cfg=None, runner_ctx={"run_config": {}})

    assert tracker is None


def test_wandb_config_includes_resolved_context(monkeypatch):
    created_runs = []

    class DummyRun:
        def __init__(self, kwargs):
            self.kwargs = kwargs
            self.summary = {}

        def log(self, payload):
            pass

        def finish(self):
            pass

    def fake_init(**kwargs):
        dummy = DummyRun(kwargs)
        created_runs.append(dummy)
        return dummy

    fake_module = types.SimpleNamespace(init=fake_init)
    monkeypatch.setitem(rs.sys.modules, "wandb", fake_module)

    cfg = rs.WandbConfig(enabled=True, project="proj", entity="ent", tags=["tag"], mode="offline")
    run_cfg = {
        "run_id": "run123",
        "labels": {},
        "container": {"image": "img", "tag": "t1"},
    }
    runner_ctx = {
        "run_config": {"concurrency": 2},
        "container_runtime": {"mounts": ["/mnt:/mnt"], "env": ["A=B"], "runner_entry": ["python3", "-m", "profile.cli.run_single_run"]},
        "runner_cfg": {"container_runtime": {"mounts": ["/mnt:/mnt"], "env": ["A=B"]}},
    }
    tracker = rs._start_wandb_run(cfg, "comp", run_cfg, profiler_cfg={"name": "level_a"}, runner_ctx=runner_ctx)

    assert tracker is not None
    recorded_cfg = created_runs[0].kwargs["config"]
    assert recorded_cfg["container"]["image"] == "img"
    assert recorded_cfg["container_runtime"]["mounts"] == ["/mnt:/mnt"]


def test_reset_run_dirs_removes_existing(tmp_path):
    raw_dir = tmp_path / "raw" / "esmfold" / "level_a" / "run_1"
    prof_dir = raw_dir
    raw_dir.mkdir(parents=True)
    (raw_dir / "profile_SUCCESS").write_text("success")

    rs._reset_run_dirs({"PROF_DIR": str(prof_dir), "PROF_DATA_DIR": str(raw_dir)})

    assert not prof_dir.exists()
    assert not raw_dir.exists()


def test_payload_log_failure_hint_flags_oom(tmp_path):
    log_path = tmp_path / "payload.log"
    log_path.write_text("RuntimeError: CUDA out of memory while running inference.\nFailed to predict batch of size 4")

    hint = _shared._payload_log_failure_hint(log_path)
    assert hint is not None
    assert "out of memory" in hint.lower()


def test_wandb_profiler_cfg_resolves_run_id(monkeypatch):
    created_runs = []

    class DummyRun:
        def __init__(self, kwargs):
            self.kwargs = kwargs
            self.summary = {}

        def log(self, payload):
            pass

        def finish(self):
            pass

    def fake_init(**kwargs):
        dummy = DummyRun(kwargs)
        created_runs.append(dummy)
        return dummy

    fake_module = types.SimpleNamespace(init=fake_init)
    monkeypatch.setitem(rs.sys.modules, "wandb", fake_module)

    cfg = rs.WandbConfig(enabled=True, project="proj", entity="ent", tags=["tag"], mode="offline")
    run_cfg = {"run_id": "run456", "labels": {}}
    runner_ctx = {"run_config": {}}
    profiler_cfg = OmegaConf.create({"output_filename": "${runner.run_id}.nsys-rep"})

    tracker = rs._start_wandb_run(cfg, "comp", run_cfg, profiler_cfg=profiler_cfg, runner_ctx=runner_ctx)

    assert tracker is not None
    recorded_cfg = created_runs[0].kwargs["config"]
    assert recorded_cfg["profiler_cfg"]["output_filename"] == "run456.nsys-rep"


def test_persist_run_payload_includes_wandb_and_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(rs, "REPO_ROOT", tmp_path)
    cache_dir = tmp_path / ".hydra_cache"
    monkeypatch.setattr(_shared, "resolve_profiler_artifacts", lambda cfg, profiler_cfg, ctx=None: [Path("artifact.txt")])

    runner = rs.ProfilingRunner(
        components=["dummy"],
        config_cache_dir=cache_dir,
        wandb=rs.WandbConfig(enabled=True, project="proj-test", tags=["tag1"]),
    )
    spec = DummySpec(container=ContainerConfig(image="dummy"))
    spec.exec_helper = None
    run_cfg = sample_run(tmp_path, meta={"scaffold": "s", "ligand": "l", "quality": "high"}, run_idx=1)

    payload_path, _ = runner._persist_run_payload(spec, run_cfg, "run-id")
    data = payload_path.read_text()
    assert "artifact.txt" in data
    assert "wandb" in data
    payload_path.unlink(missing_ok=True)


def test_summarize_process_error_handles_bytes_output():
    exc = subprocess.TimeoutExpired(cmd=["docker", "run"], timeout=5, output=b"line1\nline2", stderr=b"errline")
    summary = rs._summarize_process_error(exc)
    assert "line2" in summary
    assert "errline" in summary


def test_container_exit_code_124_is_classified_as_timeout(monkeypatch, tmp_path):
    recorded = []

    def fake_run_component_in_container(_spec, **_kwargs):
        raise subprocess.CalledProcessError(124, ["docker", "run"])

    runner = rs.ProfilingRunner(
        components=["dummy"],
        config_cache_dir=tmp_path / ".hydra_cache",
        show_progress=False,
    )
    runner.dry_run = False
    runner._profiler_cfg = None

    runner._record_run_result = lambda **kwargs: recorded.append(kwargs)  # type: ignore[method-assign]
    monkeypatch.setattr(rs, "run_component_in_container", fake_run_component_in_container)

    spec = DummySpec(container=ContainerConfig(image="dummy"))
    cfg = sample_run(tmp_path, run_idx=1)

    runner.run_runs_in_container(spec, [cfg])

    assert recorded
    assert recorded[0]["status"] == "timeout"
    assert recorded[0]["exit_code"] == 124


def test_smoke_report_counts_timeout_as_success(tmp_path):
    report = rs.SmokeReport(
        report_root=tmp_path,
        timestamp="ts",
        profiler=None,
        count_timeout_as_success=True,
    )
    rec = rs.SmokeRecord(
        component="comp",
        run_id="rid",
        status="timeout",
        start_time="0",
        end_time="1",
        meta={},
        profiler=None,
        exit_code=None,
        error_summary=None,
    )
    report.add(rec)
    assert report.is_success(rec)
    json_path, md_path = report.write()
    assert json_path.exists()
    content = md_path.read_text()
    assert "1/1 success" in content
    assert "treated as success" in content


def test_smoke_mode_can_use_full_combos(monkeypatch, tmp_path):
    build_runs_called = []
    build_smoke_called = []

    class FakeSpec:
        name = "fake"
        container = None
        run_gpu = False

        def build_runs(self, *, timestamp, repeats):
            build_runs_called.append((timestamp, repeats))
            return [sample_run(tmp_path, run_idx=1)]

        def build_smoke_runs(self, *, timestamp, repeats):
            build_smoke_called.append((timestamp, repeats))
            return []

    runner = rs.ProfilingRunner(components=["fake"], mode="smoke", repeats=1, smoke=rs.SmokeConfig(report_dir=tmp_path))
    runner._profiler_cfg = OmegaConf.create(
        {
            "_target_": "profile.profilers.strategies.LevelATelemetryStrategy",
            "name": "level_a",
            "requires_nsight": False,
        }
    )
    runner.dry_run = True
    monkeypatch.setattr(rs, "available_components", lambda: ["fake"])
    monkeypatch.setattr(runner, "_get_component_spec", lambda name: FakeSpec())

    exit_code = runner.run()
    assert exit_code == 0
    assert build_runs_called
    assert not build_smoke_called


def test_container_runner_entry_override(monkeypatch, tmp_path):
    captured = {}

    class FakeContainer(ContainerConfig):
        def __init__(self):
            super().__init__(image="dummy", runner_entry=["python", "-m", "custom"])

    class FakeSpec:
        name = "fake"
        container = FakeContainer()
        run_gpu = False

        def build_runs(self, *, timestamp, repeats):
            return [sample_run(tmp_path, run_idx=1)]

        def build_smoke_runs(self, *, timestamp, repeats):
            return [sample_run(tmp_path, run_idx=1)]

    def fake_run_component_in_container(spec, **kwargs):
        captured["runner_entry"] = kwargs.get("runner_entry")
        raise SystemExit

    runner = rs.ProfilingRunner(components=["fake"], mode="smoke", repeats=1, smoke=rs.SmokeConfig(report_dir=tmp_path))
    runner._profiler_cfg = OmegaConf.create(
        {
            "_target_": "profile.profilers.strategies.LevelATelemetryStrategy",
            "name": "level_a",
            "requires_nsight": False,
        }
    )
    runner.dry_run = False
    monkeypatch.setattr(rs, "available_components", lambda: ["fake"])
    monkeypatch.setattr(runner, "_get_component_spec", lambda name: FakeSpec())
    monkeypatch.setattr(rs, "run_component_in_container", fake_run_component_in_container)

    with pytest.raises(SystemExit):
        runner.run()

    assert captured["runner_entry"] == ("python", "-m", "custom", "--payload", ANY)


def test_outputs_are_scoped_per_run(monkeypatch, tmp_path):
    base = tmp_path / "scratch" / "dummy"

    monkeypatch.setattr(rs, "_component_output_base", lambda name: base)

    cfg = sample_run(tmp_path, run_idx=1)
    cfg["CMD"] = [f"{base}/foo", "unchanged"]
    cfg["PARAMS"] = [f"--out={base}/bar", "--keep"]
    cfg["PREP_COMMANDS"] = [f"rm -rf {base}/output", f"mkdir -p {base}/output"]

    scoped = rs._scope_output_paths(cfg, "dummy", "dummy-run")

    assert scoped == base / "runs" / "dummy-run"
    assert str(scoped) in cfg["CMD"][0]
    assert str(scoped) in cfg["PARAMS"][0]
    assert all(str(scoped) in cmd for cmd in cfg["PREP_COMMANDS"])


def test_telemetry_strategy_requires_psutil(monkeypatch, tmp_path):
    monkeypatch.setattr(strategies, "psutil", None)
    monitor = strategies.TelemetryMonitor(interval=0.1, output_file=tmp_path / "telemetry.csv", gpu_query="index,name")
    try:
        try:
            monitor._collect_row()  # type: ignore[attr-defined]
        except RuntimeError as exc:
            assert "psutil" in str(exc)
        else:  # pragma: no cover - should not reach
            assert False, "Expected RuntimeError when psutil is missing"
    finally:
        monitor.stop()


def test_container_paths_use_repo_root(monkeypatch, tmp_path):
    monkeypatch.setattr(rs, "REPO_ROOT", tmp_path / "outer")
    runner = rs.ProfilingRunner(
        components=["dummy"],
        repo_root=tmp_path,
        config_cache_dir=tmp_path / ".hydra_cache",
    )
    runner.container_workdir = "/workspace"
    spec = DummySpec(container=ContainerConfig(image="dummy"))
    cfg = sample_run(tmp_path, run_idx=1)
    host_path, container_path = runner._persist_run_payload(spec, cfg, "rid")
    assert Path(container_path).as_posix().startswith("/workspace/.hydra_cache/")
    host_path.unlink(missing_ok=True)
