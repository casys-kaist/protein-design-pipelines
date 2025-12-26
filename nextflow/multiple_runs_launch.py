#!/usr/bin/env python3
"""Launch denovo fan-out experiments across fanout/collocation/GPU matrices."""

import argparse
import csv
import datetime as dt
import json
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TextIO

from tqdm import tqdm

NEXTFLOW_DIR = Path(__file__).resolve().parent
REPO_ROOT = NEXTFLOW_DIR.parent

# ========= CONFIG =========
BASE_INPUT = "denovo_1n0r_3prs"
RESULTS_ROOT = (REPO_ROOT / "results" / "denovo" / BASE_INPUT).resolve()
MANIFEST_PATH = RESULTS_ROOT / "multi_run_manifest_v2.jsonl"
FANOUT_LEVELS = [1, 2, 3]
COLLOCATE_OPTIONS = [False, True]  # False -> override to 24, True -> use defaults
GPU_COUNTS = [1, 4]
DEFAULT_REPEATS = 3
MAX_ATTEMPTS = 2
RETRY_DELAY_SEC = 120
COOLDOWN_SEC = 20
DEFAULT_STRUCT_MODE = "protenix"
DEFAULT_DOCK_MODE = "vina_gpu"
DEFAULT_WORK_DIR = (NEXTFLOW_DIR / "work").resolve()
DEFAULT_INPUT_PATH = (NEXTFLOW_DIR / "samplesheet" / "handpicked" / "1n0r_3prs.csv").resolve()
CONFIG_CACHE_DIR = NEXTFLOW_DIR / ".multi_run_configs"
FIXED_MMSEQS_BATCH_SIZE = 10
BLOCKER_NAMESPACE = "nextflow"
BLOCKER_NAME_PREFIX = "gpu-blocker"
DEFAULT_GPU_CAPACITY = 4
GPU_MONITOR_FILENAME = "gpu_metrics.csv"
# 100 ms sampling cadence for GPU telemetry (uses nvidia-smi --loop-ms under the hood)
GPU_MONITOR_INTERVAL_SEC = 0.1
NVIDIA_SMI_AVAILABLE = shutil.which("nvidia-smi") is not None
GPU_MONITOR_NOTICE_EMITTED = False
NF_POD_PREFIX = "nf-"
TRACE_SUCCESS_STATUSES = {"COMPLETED", "CACHED", "SKIPPED"}
# Canonical combo set for the trimmed OBS#2 sweep discussed with the
# experimentation team.
TRIMMED_OBS2_COMBOS = [
    (1, False, 1),
    (2, False, 1),
    (3, False, 1),
    (2, True, 1),
    (3, True, 1),
    (2, True, 4),
    (3, True, 4),
]
# ==========================


@dataclass
class RunPlan:
    fanout_level: int
    collocate: bool
    gpu_count: int
    run_index: int
    outdir: Path


@dataclass
class ManifestEntry:
    fanout_level: int
    collocate: bool
    gpu_count: int
    run_index: int
    outdir: str
    status: str
    started_at: str
    finished_at: Optional[str] = None
    trace_file: Optional[str] = None
    message: Optional[str] = None


class GPUUtilizationMonitor:
    """Capture periodic GPU metrics via ``nvidia-smi`` while a run is active."""

    QUERY_FIELDS = [
        "timestamp",
        "index",
        "uuid",
        "name",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
    ]
    HEADER = QUERY_FIELDS

    def __init__(self, output_path: Path, interval_sec: float = GPU_MONITOR_INTERVAL_SEC) -> None:
        self.output_path = output_path
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._file_handle: Optional[TextIO] = None
        self._active = False
        self._error: Optional[str] = None

    @staticmethod
    def _nvidia_smi_cmd() -> List[str]:
        return [
            "nvidia-smi",
            f"--query-gpu={','.join(GPUUtilizationMonitor.QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]

    def start(self) -> bool:
        if not NVIDIA_SMI_AVAILABLE:
            return False

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.output_path.exists() or self.output_path.stat().st_size == 0
        self._file_handle = self.output_path.open("a", buffering=1)
        if write_header:
            self._file_handle.write(",".join(self.HEADER) + "\n")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)
        self._thread.start()
        self._active = True
        return True

    def stop(self) -> None:
        if not self._active:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.interval_sec + 1)
        if self._file_handle:
            self._file_handle.flush()
            self._file_handle.close()
            self._file_handle = None
        self._thread = None
        self._active = False

    def _run(self) -> None:
        if self.interval_sec < 1.0:
            self._run_streaming()
            return

        while not self._stop_event.is_set():
            start_time = time.time()
            try:
                result = subprocess.run(
                    self._nvidia_smi_cmd(),
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                self._error = str(exc)
                print(f"GPU monitor stopped: {exc}")
                break

            if self._file_handle:
                lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                for line in lines:
                    self._file_handle.write(line + "\n")
                self._file_handle.flush()

            elapsed = time.time() - start_time
            wait_time = max(0.0, self.interval_sec - elapsed)
            if self._stop_event.wait(wait_time):
                break

    def _run_streaming(self) -> None:
        interval_ms = max(1, int(self.interval_sec * 1000))
        cmd = self._nvidia_smi_cmd() + [f"--loop-ms={interval_ms}"]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (OSError, ValueError) as exc:
            self._error = str(exc)
            print(f"GPU monitor failed to start: {exc}")
            return

        try:
            while not self._stop_event.is_set():
                if proc.stdout is None:
                    break
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    continue
                line = line.strip()
                if not line:
                    continue
                if self._file_handle:
                    self._file_handle.write(line + "\n")
                    self._file_handle.flush()
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        finally:
            if proc.stdout:
                proc.stdout.close()
            stderr_text = None
            if proc.stderr:
                try:
                    stderr_text = proc.stderr.read()
                except Exception:
                    stderr_text = None
                finally:
                    proc.stderr.close()
            if stderr_text and not self._error:
                self._error = stderr_text.strip()

    @property
    def error(self) -> Optional[str]:
        return self._error

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Number of repeats per combination (default: %(default)s).",
    )
    parser.add_argument(
        "--stub", action="store_true",
        help="Enable Nextflow stub run (skips manifest bookkeeping and GPU blockers).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip runs that already have successful Nextflow trace logs in their outdir. "
            "Useful for resuming interrupted sweeps."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Pass -resume to Nextflow to reuse cached results from prior runs.",
    )
    parser.add_argument(
        "--stub-outdir", type=Path, default=(REPO_ROOT / "results" / "stub_runs").resolve(),
        help="Root outdir when running in stub mode (default: %(default)s).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Samplesheet CSV to feed into the pipeline.",
    )
    parser.add_argument(
        "--fanout-levels",
        type=int,
        nargs="+",
        help="Subset of fanout levels to run (default: all levels %s)."
        % FANOUT_LEVELS,
    )
    parser.add_argument(
        "--collocate-options",
        nargs="+",
        choices=["exclusive", "collocate", "true", "false"],
        help="Subset of collocate modes to run (default: both). Accepts 'exclusive'/'false' or 'collocate'/'true'.",
    )
    parser.add_argument(
        "--gpu-counts",
        type=int,
        nargs="+",
        help="Subset of GPU counts to run (default: %s)." % GPU_COUNTS,
    )
    parser.add_argument(
        "--combo-preset",
        choices=["all", "trimmed_obs2"],
        default="all",
        help=(
            "Optional preset defining the fanout/collocate/GPU combinations to run. "
            "'trimmed_obs2' uses the 7 unique combos requested for the OBS#2 sweep."
        ),
    )
    parser.add_argument(
        "--gpu-capacity",
        type=int,
        default=DEFAULT_GPU_CAPACITY,
        help=(
            "Total GPUs available in the target cluster when deploying blocker pods "
            "(default: %(default)s)"
        ),
    )
    return parser.parse_args()


def load_manifest() -> Dict[Tuple[int, bool, int, int], ManifestEntry]:
    records: Dict[Tuple[int, bool, int, int], ManifestEntry] = {}
    if not MANIFEST_PATH.exists():
        return records

    with MANIFEST_PATH.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            key = (
                int(raw["fanout_level"]),
                bool(raw["collocate"]),
                int(raw["gpu_count"]),
                int(raw["run_index"]),
            )
            records[key] = ManifestEntry(
                fanout_level=int(raw["fanout_level"]),
                collocate=bool(raw["collocate"]),
                gpu_count=int(raw["gpu_count"]),
                run_index=int(raw["run_index"]),
                outdir=raw["outdir"],
                status=raw["status"],
                started_at=raw.get("started_at"),
                finished_at=raw.get("finished_at"),
                trace_file=raw.get("trace_file"),
                message=raw.get("message"),
            )
    return records


def append_manifest(entry: ManifestEntry) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("a") as handle:
        payload = asdict(entry)
        json.dump(payload, handle)
        handle.write("\n")


def cleanup_outdir(path: Path) -> None:
    if path.exists():
        print(f"Cleaning existing outdir: {path}")
        shutil.rmtree(path)


def allocate_run_indices(combo_root: Path, repeats: int, include_existing: bool = False) -> List[int]:
    """
    Return run indices for the next ``repeats`` runs.

    When ``include_existing`` is True, include existing run indices so we can
    validate whether they completed successfully.
    """
    if repeats <= 0:
        return []

    if include_existing:
        return list(range(1, repeats + 1))

    indices: List[int] = []
    candidate = 1
    while len(indices) < repeats:
        candidate_path = combo_root / f"run_{candidate}"
        if not candidate_path.exists():
            indices.append(candidate)
        candidate += 1
    return indices


def prepare_config(collocate: bool) -> Optional[Path]:
    if collocate:
        return None

    CONFIG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIG_CACHE_DIR / "no_collocate.config"
    base_config = (NEXTFLOW_DIR / "nextflow.config").resolve()
    config_path.write_text(
        """
includeConfig '%s'

process {
    withLabel:gpu {
        accelerator = [ request: 24, limit: 24, type: 'aliyun.com/gpu-mem' ]
    }
}
""".strip()
        % base_config.as_posix()
    )
    return config_path


def build_nextflow_cmd(
    plan: RunPlan,
    input_csv: Path,
    stub: bool,
    resume: bool,
    config_path: Optional[Path],
) -> List[str]:
    fan = plan.fanout_level
    cmd: List[str] = [
        "nextflow",
        "run",
        "main.nf",
        "-profile",
        "k8s",
        "-work-dir",
        str(DEFAULT_WORK_DIR),
    ]

    if resume:
        cmd.append("-resume")

    if config_path is not None:
        cmd.extend(["-c", str(config_path)])

    cmd.extend(
        [
            "--pipeline",
            "denovo_design_ligand",
            "--input",
            str(input_csv),
            "--outdir",
            str(plan.outdir.resolve()),
            "--structure_prediction_mode",
            DEFAULT_STRUCT_MODE,
            "--docking_mode",
            DEFAULT_DOCK_MODE,
            "--rfdiffusion_num_designs",
            str(fan),
            "--proteinmpnn_num_seq_per_target",
            str(fan),
            "--protenix_num_samples",
            str(fan),
            "--esm_num_variants",
            str(fan),
            "--mmseqs2_batch_size",
            str(FIXED_MMSEQS_BATCH_SIZE),
        ]
    )

    if stub:
        cmd.append("-stub-run")

    return cmd


def run_with_retries(cmd: List[str], monitor_path: Optional[Path] = None) -> Tuple[bool, Optional[str]]:
    global GPU_MONITOR_NOTICE_EMITTED

    last_error: Optional[str] = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        monitor: Optional[GPUUtilizationMonitor] = (
            GPUUtilizationMonitor(monitor_path) if monitor_path else None
        )
        try:
            if monitor:
                started = monitor.start()
                if (
                    not started
                    and not NVIDIA_SMI_AVAILABLE
                    and not GPU_MONITOR_NOTICE_EMITTED
                ):
                    print("Skipping GPU monitoring: nvidia-smi not found on PATH.")
                    GPU_MONITOR_NOTICE_EMITTED = True

            subprocess.run(cmd, check=True, cwd=str(NEXTFLOW_DIR))
            success = True
            last_error = None
        except subprocess.CalledProcessError as exc:
            success = False
            last_error = str(exc)
            print(f"Run failed on attempt {attempt}: {last_error}")
        finally:
            if monitor:
                monitor.stop()
                if monitor.error and not GPU_MONITOR_NOTICE_EMITTED:
                    print(f"GPU monitoring ended with error: {monitor.error}")
                    GPU_MONITOR_NOTICE_EMITTED = True

        if success:
            time.sleep(COOLDOWN_SEC)
            return True, None

        if attempt < MAX_ATTEMPTS:
            print(f"Waiting {RETRY_DELAY_SEC}s before retry...")
            time.sleep(RETRY_DELAY_SEC)

    return False, last_error or "Unknown failure"


def find_trace_file(run_outdir: Path) -> Optional[Path]:
    trace_dir = run_outdir / "pipeline_info"
    if not trace_dir.exists():
        return None
    traces = sorted(trace_dir.glob("execution_trace_*.txt"))
    return traces[-1] if traces else None


def trace_is_successful(trace_path: Path) -> bool:
    try:
        with trace_path.open() as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames or "status" not in reader.fieldnames:
                return False
            any_rows = False
            for row in reader:
                any_rows = True
                status = (row.get("status") or "").strip().upper()
                if not status or status not in TRACE_SUCCESS_STATUSES:
                    return False
            return any_rows
    except OSError as exc:
        print(f"Unable to read trace file {trace_path}: {exc}")
        return False


def run_has_success_log(run_outdir: Path) -> bool:
    trace_path = find_trace_file(run_outdir)
    if not trace_path:
        return False
    return trace_is_successful(trace_path)


def create_blocker_pod_yaml(name: str) -> str:
    return f"""
apiVersion: v1
kind: Pod
metadata:
  name: {name}
  namespace: {BLOCKER_NAMESPACE}
spec:
  restartPolicy: Never
  containers:
  - name: blocker
    image: protenix:latest
    command: ["/bin/bash", "-c", "sleep infinity"]
    resources:
      requests:
        aliyun.com/gpu-mem: 24
        memory: "0Gi"
      limits:
        aliyun.com/gpu-mem: 24
        memory: "0Gi"
""".strip()


def set_blocker_state(target_count: int, state: Dict[str, int]) -> None:
    """Ensure the desired number of blocker pods are present."""

    target_count = max(0, int(target_count))
    current = state.get("count", 0)
    if target_count == current:
        return

    if target_count == 0:
        if current > 0:
            print("Removing GPU blocker pods...")
            for idx in range(1, current + 1):
                _delete_blocker(idx)
        state["count"] = 0
        return

    if current == 0:
        print(f"Applying {target_count} GPU blocker pod(s) (gpu-mem:24 each)...")
        for idx in range(1, target_count + 1):
            _create_blocker(idx)
        state["count"] = target_count
        return

    if target_count > current:
        print(f"Scaling GPU blocker pods up to {target_count}...")
        for idx in range(current + 1, target_count + 1):
            _create_blocker(idx)
    else:
        print(f"Scaling GPU blocker pods down to {target_count}...")
        for idx in range(target_count + 1, current + 1):
            _delete_blocker(idx)

    state["count"] = target_count


def _create_blocker(idx: int) -> None:
    name = f"{BLOCKER_NAME_PREFIX}-{idx}"
    yaml_body = create_blocker_pod_yaml(name)
    subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        check=True,
        cwd=str(NEXTFLOW_DIR),
        input=yaml_body.encode("utf-8"),
    )


def _delete_blocker(idx: int) -> None:
    name = f"{BLOCKER_NAME_PREFIX}-{idx}"
    try:
        subprocess.run(
            [
                "kubectl",
                "delete",
                "pod",
                name,
                "-n",
                BLOCKER_NAMESPACE,
                "--ignore-not-found=true",
            ],
            check=True,
            cwd=str(NEXTFLOW_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        pass


def cleanup_nf_pods(namespace: str = BLOCKER_NAMESPACE, prefix: str = NF_POD_PREFIX) -> None:
    """Delete lingering Nextflow pods (names starting with ``prefix``)."""

    if shutil.which("kubectl") is None:
        print("Skipping Nextflow pod cleanup: kubectl not found on PATH.")
        return

    try:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", namespace, "-o", "json"],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(NEXTFLOW_DIR),
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"Failed to list pods for cleanup: {exc}")
        return

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(f"Unable to parse pod list JSON: {exc}")
        return

    pods = [
        item.get("metadata", {}).get("name", "")
        for item in payload.get("items", [])
        if item.get("metadata", {}).get("name", "").startswith(prefix)
    ]

    if not pods:
        return

    print(f"Deleting lingering Nextflow pods: {', '.join(pods)}")
    try:
        subprocess.run(
            [
                "kubectl",
                "delete",
                "pod",
                *pods,
                "-n",
                namespace,
                "--ignore-not-found=true",
            ],
            check=True,
            cwd=str(NEXTFLOW_DIR),
        )
    except subprocess.CalledProcessError as exc:
        print(f"Pod cleanup encountered errors: {exc}")


def format_combo_label(fanout: int, collocate: bool, gpu_count: int) -> str:
    collocate_label = "collocate" if collocate else "exclusive"
    return f"fanout-{fanout}_collocate-{collocate_label}_gpu-{gpu_count}"


def main() -> None:
    args = parse_args()

    input_csv = args.input.resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"--input: the file or directory '{input_csv}' does not exist")

    repeats = max(1, args.repeats)
    stub_root = args.stub_outdir.resolve() if args.stub else None

    manifest = load_manifest() if not args.stub else {}
    blocker_state = {"count": 0}

    fanout_levels = args.fanout_levels or FANOUT_LEVELS

    if args.collocate_options:
        collocate_options: List[bool] = []
        seen_collocates: set = set()
        for entry in args.collocate_options:
            entry_lower = entry.lower()
            if entry_lower in ("collocate", "true", "1", "yes"):
                value = True
            elif entry_lower in ("exclusive", "false", "0", "no"):
                value = False
            else:
                raise ValueError(
                    f"Unsupported collocate option '{entry}'. Use one of: exclusive, collocate, true, false."
                )
            if value not in seen_collocates:
                collocate_options.append(value)
                seen_collocates.add(value)
    else:
        collocate_options = COLLOCATE_OPTIONS

    gpu_counts = args.gpu_counts or GPU_COUNTS

    if args.combo_preset == "trimmed_obs2":
        base_combos = TRIMMED_OBS2_COMBOS
        combos = [
            combo
            for combo in base_combos
            if combo[0] in fanout_levels
            and combo[1] in collocate_options
            and combo[2] in gpu_counts
        ]
    else:
        combos = [
            (fanout, collocate, gpu)
            for fanout in fanout_levels
            for collocate in collocate_options
            for gpu in gpu_counts
        ]

    if not combos:
        print("No run combinations selected; exiting.")
        return

    total_combos = len(combos)
    combo_pbar = tqdm(total=total_combos, desc="Combos", unit="combo")
    combo_config_cache: Dict[bool, Optional[Path]] = {}

    for fanout, collocate, gpu_count in combos:
        combo_label = format_combo_label(fanout, collocate, gpu_count)
        combo_pbar.set_postfix(
            {"fanout": fanout, "collocate": collocate, "gpu": gpu_count}
        )
        combo_root = (
            (stub_root if args.stub else RESULTS_ROOT)
            / f"fanout_{fanout}"
            / ("collocate" if collocate else "exclusive")
            / f"gpu_{gpu_count}"
        ).resolve()

        run_pbar = tqdm(
            total=repeats,
            desc=f"{combo_label}",
            unit="run",
            leave=False,
        )

        run_indices = allocate_run_indices(combo_root, repeats, include_existing=args.skip_existing)
        pending: List[RunPlan] = []
        for run_idx in run_indices:
            outdir = combo_root / f"run_{run_idx}"
            plan = RunPlan(
                fanout_level=fanout,
                collocate=collocate,
                gpu_count=gpu_count,
                run_index=run_idx,
                outdir=outdir,
            )

            key = (fanout, collocate, gpu_count, run_idx)
            entry = manifest.get(key)
            if (
                not args.stub
                and entry
                and entry.status == "success"
                and Path(entry.outdir).exists()
            ):
                print(f"Skipping completed run: {combo_label} repeat {run_idx}")
                run_pbar.update(1)
                continue
            if (
                not args.stub
                and args.skip_existing
                and outdir.exists()
                and run_has_success_log(outdir)
            ):
                print(f"Skipping existing successful run: {combo_label} repeat {run_idx}")
                run_pbar.update(1)
                continue
            pending.append(plan)

        if not pending:
            run_pbar.close()
            combo_pbar.update(1)
            continue

        if not args.stub:
            if collocate not in combo_config_cache:
                combo_config_cache[collocate] = prepare_config(collocate)
            config_path = combo_config_cache[collocate]
            blockers_needed = max(0, args.gpu_capacity - gpu_count)
            set_blocker_state(blockers_needed, blocker_state)
        else:
            config_path = None

        for plan in pending:
            plan.outdir.parent.mkdir(parents=True, exist_ok=True)
            cleanup_outdir(plan.outdir)
            plan.outdir.mkdir(parents=True, exist_ok=True)

            if not args.stub:
                cleanup_nf_pods()

            cmd = build_nextflow_cmd(plan, input_csv, args.stub, args.resume, config_path)
            monitor_path = plan.outdir / GPU_MONITOR_FILENAME
            started = dt.datetime.utcnow().isoformat()
            success, message = run_with_retries(cmd, monitor_path=monitor_path)
            finished = dt.datetime.utcnow().isoformat()

            if args.stub:
                status = "success" if success else "failed"
                print(f"Stub run {status}: {combo_label} repeat {plan.run_index}")
                run_pbar.update(1)
                continue

            if not success:
                cleanup_outdir(plan.outdir)
                append_manifest(
                    ManifestEntry(
                        fanout_level=plan.fanout_level,
                        collocate=plan.collocate,
                        gpu_count=plan.gpu_count,
                        run_index=plan.run_index,
                        outdir=str(plan.outdir),
                        status="failed",
                        started_at=started,
                        finished_at=finished,
                        message=message,
                    )
                )
                run_pbar.update(1)
                continue

            trace_file = find_trace_file(plan.outdir)
            append_manifest(
                ManifestEntry(
                    fanout_level=plan.fanout_level,
                    collocate=plan.collocate,
                    gpu_count=plan.gpu_count,
                    run_index=plan.run_index,
                    outdir=str(plan.outdir),
                    status="success",
                    started_at=started,
                    finished_at=finished,
                    trace_file=str(trace_file) if trace_file else None,
                )
            )
            run_pbar.update(1)

        run_pbar.close()
        combo_pbar.update(1)

    combo_pbar.close()

    if not args.stub:
        set_blocker_state(0, blocker_state)


if __name__ == "__main__":
    main()
