import csv
import io
import os
import re
import shutil
import sqlite3
import subprocess
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _normalize_sections(value: Optional[Sequence[str]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    else:
        items = list(value)
    return [str(item).strip() for item in items if str(item).strip()]


class NoCudaKernelsError(RuntimeError):
    pass


def _resolve_ncu_binary() -> str:
    override = os.environ.get("NCU_BIN")
    if override:
        return override
    for candidate in (
        Path("/usr/local/bin/ncu"),
        Path("/opt/nvidia/nsight-compute/current/ncu"),
    ):
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    root = Path("/opt/nvidia/nsight-compute")
    if root.exists():
        versioned: List[Tuple[Tuple[int, ...], Path]] = []
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if child.name == "current":
                continue
            parts = child.name.split(".")
            if not parts or not all(part.isdigit() for part in parts):
                continue
            ncu_path = child / "ncu"
            if ncu_path.exists() and os.access(ncu_path, os.X_OK):
                version = tuple(int(part) for part in parts)
                versioned.append((version, ncu_path))
        if versioned:
            versioned.sort()
            return str(versioned[-1][1])
    resolved = shutil.which("ncu")
    return resolved or "ncu"


@dataclass
class ProfilePaths:
    nsys_rep: Path
    sqlite_db: Path
    ncu_dir: Path
    kernel_csv: Path
    summary_csv: Path
    roofline_rep: Path


@dataclass
class ProfileConfig:
    cmd: Sequence[str]
    params: Sequence[str]
    prof_dir: Path
    prof_data_dir: Path
    basename: str
    start_index: int = 1
    prep_commands: Sequence[str] = field(default_factory=list)
    require_existing_nsys_rep: bool = True
    create_roofline_report: bool = False
    env_overrides: Dict[str, str] = field(default_factory=dict)
    trace_path: Optional[Path] = None
    ncu_sections: Optional[Sequence[str]] = None

    def paths(self) -> ProfilePaths:
        base_trace = Path(self.trace_path) if self.trace_path else Path(self.prof_data_dir) / f"{self.basename}_trace.nsys-rep"
        return ProfilePaths(
            nsys_rep=base_trace,
            sqlite_db=base_trace.with_suffix(".sqlite"),
            ncu_dir=Path(self.prof_data_dir) / f"{self.basename}_ncu",
            # Persist kernel/summary artifacts alongside raw profiler data so we avoid
            # legacy timestamped result dirs.
            kernel_csv=Path(self.prof_data_dir) / f"{self.basename}_kernel_summary.csv",
            summary_csv=Path(self.prof_data_dir) / f"{self.basename}_ncu_summary.csv",
            roofline_rep=Path(self.prof_data_dir) / f"{self.basename}_roofline.ncu-rep",
        )

    def cleaned_env(self) -> Dict[str, str]:
        env = deepcopy(os.environ)
        pythonpath = env.get("PYTHONPATH")
        if pythonpath:
            parts = [p for p in pythonpath.split(os.pathsep) if "profiling/src" not in p]
            if parts:
                env["PYTHONPATH"] = os.pathsep.join(parts)
            else:
                env.pop("PYTHONPATH", None)
        else:
            env.pop("PYTHONPATH", None)
        for key, value in self.env_overrides.items():
            env[str(key)] = str(value)
        return env

    def ensure_dirs(self, paths: ProfilePaths) -> None:
        Path(self.prof_dir).mkdir(parents=True, exist_ok=True)
        paths.ncu_dir.mkdir(parents=True, exist_ok=True)


def _clean_metric(value: Optional[str]) -> float:
    if value is None:
        return 0.0
    try:
        value = str(value).replace(",", "")
        if value.endswith("%"):
            value = value[:-1]
        return float(value)
    except ValueError:
        return 0.0


def _extract_metrics_from_ncu(cfg: ProfileConfig, ncu_base_path: Path, writer: csv.writer, index: int, kernel_name: str) -> None:
    requested_sections = set(_normalize_sections(cfg.ncu_sections) or ["Occupancy", "WarpStateStats"])
    ncu_bin = _resolve_ncu_binary()

    def extract_all_metrics(section: str, target_metrics: Iterable[str]) -> List[Tuple[str, str, str]]:
        csv_file = f"{ncu_base_path}.ncu-rep"
        try:
            output = subprocess.check_output(
                [ncu_bin, "--import", csv_file, "--csv", "--section", section],
                encoding="utf-8",
                env=cfg.cleaned_env(),
            )
        except subprocess.CalledProcessError:
            print(f"[!] Failed to extract section '{section}' from {csv_file}")
            return []

        reader = csv.reader(io.StringIO(output))
        lines = list(reader)
        if not lines or "Metric Name" not in lines[0]:
            print(f"[!] Section '{section}' not found or malformed in {csv_file}")
            return []

        header = lines[0]
        try:
            metric_idx = header.index("Metric Name")
            value_idx = header.index("Metric Value")
            launch_idx = header.index("ID") if "ID" in header else -1
        except ValueError:
            print(f"[!] Required columns not found in section '{section}'")
            return []

        results: List[Tuple[str, str, str]] = []
        for row in lines[1:]:
            if len(row) <= max(metric_idx, value_idx):
                continue
            name = row[metric_idx]
            if name not in target_metrics:
                continue
            value = row[value_idx]
            launch = row[launch_idx] if launch_idx >= 0 and len(row) > launch_idx else ""
            results.append((name, value, launch))
        return results

    occupancy_metrics: List[Tuple[str, str, str]] = []
    warp_metrics: List[Tuple[str, str, str]] = []

    if "Occupancy" in requested_sections:
        occupancy_metrics = extract_all_metrics(
            "Occupancy",
            [
                "Kernel Name",
                "Theoretical Occupancy",
                "Achieved Occupancy",
                "Average Active Threads Per Warp",
            ],
        )

    if "WarpStateStats" in requested_sections:
        # Nsight Compute exports use the short section name (WarpStateStats); using the
        # long display name triggers a non-zero exit on recent NCU versions.
        warp_metrics = extract_all_metrics(
            "WarpStateStats",
            [
                "Kernel Name",
                "smsp__warps_active.avg.pct_of_peak_sustained_active",
                "dram__bytes.sum",
                "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
                "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
                "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
                "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
                "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",
                "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
                "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum",
                "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum",
                "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum",
                "smsp__sass_thread_inst_executed_op_hmma_pred_on.sum",
            ],
        )

    metrics = defaultdict(dict)
    for name, value, launch in occupancy_metrics + warp_metrics:
        metrics[launch][name] = value

    for launch, values in metrics.items():
        kernel_label = values.get("Kernel Name", kernel_name)
        theo = values.get("Theoretical Occupancy", "0")
        achv = values.get("Achieved Occupancy", "0")
        avg_threads = values.get("Average Active Threads Per Warp", "0")
        dram_bytes = float(values.get("dram__bytes.sum", "0").replace(",", "")) if values.get("dram__bytes.sum") else 0.0

        fadd = _clean_metric(values.get("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"))
        fmul = _clean_metric(values.get("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"))
        ffma = _clean_metric(values.get("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"))
        sp_flops = fadd + fmul + (ffma * 2)

        dadd = _clean_metric(values.get("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum"))
        dmul = _clean_metric(values.get("smsp__sass_thread_inst_executed_op_dmul_pred_on.sum"))
        dfma = _clean_metric(values.get("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum"))
        dp_flops = dadd + dmul + (dfma * 2)

        hadd = _clean_metric(values.get("smsp__sass_thread_inst_executed_op_hadd_pred_on.sum"))
        hmul = _clean_metric(values.get("smsp__sass_thread_inst_executed_op_hmul_pred_on.sum"))
        hfma = _clean_metric(values.get("smsp__sass_thread_inst_executed_op_hfma_pred_on.sum"))
        hp_flops = hadd + hmul + (hfma * 2)

        hmma = _clean_metric(values.get("smsp__sass_thread_inst_executed_op_hmma_pred_on.sum"))
        tc_flops = hmma * 512

        total_flops = sp_flops + dp_flops + hp_flops + tc_flops
        arithmetic_intensity = total_flops / dram_bytes if dram_bytes > 0 else 0.0

        writer.writerow([
            index, kernel_label, launch, theo, achv, avg_threads,
            int(dram_bytes), int(total_flops), f"{arithmetic_intensity:.4f}",
        ])


def run_prep_commands(cfg: ProfileConfig) -> None:
    for cmd in cfg.prep_commands:
        if not str(cmd).strip():
            continue
        print(f"▶ Running prep command: {cmd}")
        subprocess.run(cmd, shell=True, check=True, env=cfg.cleaned_env())


def extract_top_kernels(cfg: ProfileConfig, paths: ProfilePaths, threshold: float = 95.0) -> List[str]:
    print("▶ Exporting .nsys-rep to SQLite...")
    sqlite_path = paths.sqlite_db
    if not sqlite_path.exists():
        print(f"  ℹ️ {sqlite_path} not found, exporting from {paths.nsys_rep}...")
        result = subprocess.run(
            ["nsys", "export", "--type", "sqlite", "--output", str(sqlite_path), str(paths.nsys_rep)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=cfg.cleaned_env(),
        )
        if result.returncode != 0:
            error = result.stderr.strip() or "nsys export failed"
            raise RuntimeError(f"Failed to export Nsight Systems trace: {error}")
    else:
        print(f"  ✅ Reusing existing SQLite DB: {sqlite_path}")

    print("▶ Querying SQLite for top GPU kernels...")
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_KERNEL'")
        if cur.fetchone() is None:
            raise NoCudaKernelsError(
                f"SQLite DB is missing CUPTI_ACTIVITY_KIND_KERNEL table: {sqlite_path}"
            )
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='StringIds'")
        if cur.fetchone() is None:
            raise NoCudaKernelsError(
                f"SQLite DB is missing StringIds table: {sqlite_path}"
            )

        query = """
        SELECT
            s.value AS KernelName,
            SUM(k.end - k.start) AS TotalTime
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        GROUP BY s.value
        ORDER BY TotalTime DESC
        """
        cur.execute(query)
        rows = cur.fetchall()
    finally:
        con.close()

    if not rows:
        raise NoCudaKernelsError("No kernel rows found in CUPTI_ACTIVITY_KIND_KERNEL.")
    total_time = sum(row[1] for row in rows)
    if total_time == 0:
        raise NoCudaKernelsError("No kernel execution time found.")

    kernels: List[str] = []
    total_percent = 0.0
    with paths.kernel_csv.open("w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Index", "Time (%)", "Name"])
        for idx, (name, time_ns) in enumerate(rows):
            percent = (time_ns / total_time) * 100.0
            total_percent += percent
            kernels.append(name)
            writer.writerow([idx + 1, round(percent, 2), name])
            if total_percent >= threshold:
                break

    print(f"✅ Extracted {len(kernels)} kernels covering {total_percent:.2f}% of GPU time.")
    return kernels


def _load_kernels_from_csv(paths: ProfilePaths) -> List[str]:
    print("✅ Kernel summary already exists, loading from CSV...")
    kernels: List[str] = []
    with paths.kernel_csv.open(newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            kernels.append(row.get("Name"))
    if not kernels:
        raise NoCudaKernelsError(f"No kernels found in {paths.kernel_csv}")
    return kernels


def run_ncu_for_kernels(cfg: ProfileConfig, paths: ProfilePaths, kernels: List[str]) -> None:
    print("▶ Running NCU per kernel...")
    ncu_sections = _normalize_sections(cfg.ncu_sections) or ["Occupancy", "WarpStateStats"]
    append_mode = paths.summary_csv.exists() and cfg.start_index > 1
    file_mode = 'a' if append_mode else 'w'
    ncu_bin = _resolve_ncu_binary()

    with paths.summary_csv.open(file_mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not append_mode:
            writer.writerow([
                "Index", "Kernel Name", "Launch",
                "Theoretical Occupancy", "Achieved Occupancy",
                "Avg. Active Threads Per Warp",
                "DRAM Bytes", "Total FLOPs", "Arithmetic Intensity (FLOPs/Byte)",
            ])

        i = cfg.start_index
        while i <= len(kernels):
            kernel_name = kernels[i - 1]
            print(f"[{i}/{len(kernels)}] {kernel_name}")
            out_base = paths.ncu_dir / f"kernel_{i}"
            rep_file = f"{out_base}.ncu-rep"
            try:
                if os.path.exists(rep_file):
                    print(f"[i] Skipping NCU for kernel {i}, .ncu-rep already exists")
                else:
                    cmd = [
                        ncu_bin,
                        "--launch-skip", "3",
                        "--launch-count", "3",
                        "--target-processes", "all",
                    ]
                    for section in ncu_sections:
                        cmd.extend(["--section", section])
                    cmd.extend([
                        "--kernel-name", kernel_name,
                        "-o", str(out_base),
                        "-f",
                        *cfg.cmd,
                        *cfg.params,
                    ])
                    subprocess.run(cmd, check=True, env=cfg.cleaned_env())

                _extract_metrics_from_ncu(cfg, out_base, writer, i, kernel_name)
                csvfile.flush()
                i += 1
            except subprocess.CalledProcessError as exc:
                print(f"[!] Error during kernel {i}: {exc}. Aborting kernel profiling.")
                raise


def run_ncu_for_roofline(cfg: ProfileConfig, paths: ProfilePaths, kernels: List[str]) -> None:
    print("\n" + "=" * 60)
    print("▶ Generating individual Roofline Analysis Report per kernel...")
    print("=" * 60)
    if not kernels:
        print("❌ No kernels to profile for roofline reports. Skipping.")
        return

    i = cfg.start_index
    ncu_bin = _resolve_ncu_binary()
    while i <= len(kernels):
        kernel_name = kernels[i - 1].strip()
        roofline_rep_path = paths.roofline_rep.with_name(f"{cfg.basename}_roofline_{kernel_name}.ncu-rep")
        output_base = roofline_rep_path.with_suffix("")
        print(f"▶▶ Generating Roofline Report for Kernel {i}: {kernel_name}")
        try:
            if roofline_rep_path.exists():
                print(f"  [i] Skipping, roofline report already exists: {roofline_rep_path}")
            else:
                cmd = [
                    ncu_bin,
                    "--set", "full",
                    "--launch-skip", "10",
                    "--launch-count", "10",
                    "--kernel-name", kernel_name,
                    "-o", str(output_base),
                    "-f",
                    *cfg.cmd,
                    *cfg.params,
                ]
                subprocess.run(cmd, check=True, env=cfg.cleaned_env())
                print(f"  ✅ Successfully created roofline report: {roofline_rep_path}")
            i += 1
        except subprocess.CalledProcessError as exc:
            print(f"  [!] Error during roofline profiling for kernel {i}: {exc}. Aborting roofline.")
            raise


def run_profile(cfg: ProfileConfig) -> None:
    paths = cfg.paths()
    cfg.ensure_dirs(paths)

    if cfg.require_existing_nsys_rep and not paths.nsys_rep.exists():
        raise RuntimeError(f"Missing Nsight Systems trace: {paths.nsys_rep}")
    if not paths.nsys_rep.exists():
        raise RuntimeError(f"NSYS trace not found: {paths.nsys_rep}")

    total_kernels: Optional[int] = None

    print(
        f"\n=== [gpu_profile:{cfg.basename}] PROF_DIR={cfg.prof_dir} "
        f"PROF_DATA_DIR={cfg.prof_data_dir} BASENAME={cfg.basename}"
    )
    print("CMD:", " ".join(cfg.cmd))
    print("PARAMS:", " ".join(cfg.params))
    print(f"NSYS_REP: {paths.nsys_rep}")
    print(f"SQLITE_DB: {paths.sqlite_db}")

    print("▶ Preparing inputs...")
    run_prep_commands(cfg)

    try:
        if paths.kernel_csv.exists():
            kernels = _load_kernels_from_csv(paths)
        else:
            kernels = extract_top_kernels(cfg, paths)
    except NoCudaKernelsError as exc:
        print(f"[WARN] {exc}")
        print("[WARN] Skipping NCU because no CUDA kernels were captured.")
        if not paths.kernel_csv.exists():
            with paths.kernel_csv.open("w", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Index", "Time (%)", "Name"])
        if not paths.summary_csv.exists():
            with paths.summary_csv.open("w", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    "Index", "Kernel Name", "Launch",
                    "Theoretical Occupancy", "Achieved Occupancy",
                    "Avg. Active Threads Per Warp",
                    "DRAM Bytes", "Total FLOPs", "Arithmetic Intensity (FLOPs/Byte)",
                ])
        return

    if not kernels:
        raise RuntimeError("Kernel list is empty after Nsight export.")
    total_kernels = len(kernels)
    print(f"▶ Profiling {total_kernels} kernels with NCU...")

    run_ncu_for_kernels(cfg, paths, kernels)
    print("✅ CSV summary saved to:", paths.summary_csv)

    if cfg.create_roofline_report:
        run_ncu_for_roofline(cfg, paths, kernels)

    print("\n✅ All profiling tasks finished.")


# ---- Legacy compatibility for callers that still set module globals ----
def _config_from_globals() -> ProfileConfig:
    required = ["CMD", "PARAMS", "PROF_DIR", "PROF_DATA_DIR", "BASENAME"]
    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError(f"Missing required globals for legacy invocation: {', '.join(missing)}")

    raw_sections = globals().get("NCU_SECTIONS")
    ncu_sections: Optional[List[str]] = None
    if raw_sections is not None:
        if isinstance(raw_sections, str):
            ncu_sections = [raw_sections]
        elif isinstance(raw_sections, (list, tuple)):
            ncu_sections = [str(item).strip() for item in raw_sections if str(item).strip()]
        else:
            value = str(raw_sections).strip()
            if value:
                ncu_sections = [value]

    return ProfileConfig(
        cmd=list(globals().get("CMD", [])),
        params=list(globals().get("PARAMS", [])),
        prof_dir=Path(globals().get("PROF_DIR")),
        prof_data_dir=Path(globals().get("PROF_DATA_DIR")),
        basename=str(globals().get("BASENAME")),
        start_index=int(globals().get("START_INDEX", 1) or 1),
        prep_commands=list(globals().get("PREP_COMMANDS", [])),
        require_existing_nsys_rep=bool(globals().get("REQUIRE_EXISTING_NSYS_REP", True)),
        create_roofline_report=bool(globals().get("CREATE_ROOFLINE_REPORT", False)),
        env_overrides=dict(globals().get("ENV_OVERRIDES", {}) or {}),
        trace_path=Path(globals()["NSYS_REP"]) if "NSYS_REP" in globals() else None,
        ncu_sections=ncu_sections,
    )


def main() -> None:
    cfg = _config_from_globals()
    run_profile(cfg)


if __name__ == "__main__":
    main()
