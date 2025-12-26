# Profiling Hydra Workspace

Canonical Hydra workspace for profiling sweeps and containerized single-run execution (Hydra-first, container-first).

```
profiling/
├── configs/           # Hydra configs (runner, profiler, notifier, experiments, components)
├── scripts/           # Helpers (e.g., run_in_container.py)
├── src/profile/       # Profiling package (CLI, components, profilers, utils)
│   ├── cli/           # Entrypoints: run_sweeps.py (host), run_single_run.py (in-container)
│   ├── components/    # Component specs/helpers
│   ├── profilers/     # Strategy-based telemetry/Nsight
│   └── utils/         # Layout, samples, fingerprinting, etc.
├── secret.yaml        # (gitignored) notifier tokens; copy from secret.example.yaml
└── DESIGN.md          # Architecture/refactor notes
```

## Flow (host → container)
- Host (`run_sweeps.py`): Hydra composes configs, builds the run matrix (component × sample_id/scaffold × quality × repeat), and for each run writes a compact payload into `.hydra_cache/` then launches `scripts/run_in_container.py` with `--payload <path>` for that run.
- Container (`run_single_run.py`): Loads the payload directly (skipping a second Hydra composition), resolves profiler artifacts, and executes the profiler strategy.
- Default container entry: `python -m profile.cli.run_single_run`.

## Quick start (from repo root or profiling/)
```bash
# list registered components
python profiling/src/profile/cli/run_sweeps.py runner.show_components=true

# smoke mode using a small experiment set
python profiling/src/profile/cli/run_sweeps.py runner.mode=smoke experiments=minimal profiler=level_a
python profiling/src/profile/cli/run_sweeps.py runner.mode=smoke experiments=all profiler=level_a

# default experiments with Level A telemetry
python profiling/src/profile/cli/run_sweeps.py profiler=level_a experiments=minimal runner.skip_existing=false
python profiling/src/profile/cli/run_sweeps.py profiler=level_b experiments=minimal runner.skip_existing=false
python profiling/src/profile/cli/run_sweeps.py profiler=level_c experiments=minimal runner.skip_existing=false

python profiling/src/profile/cli/run_sweeps.py profiler=level_a
python profiling/src/profile/cli/run_sweeps.py profiler=level_b
python profiling/src/profile/cli/run_sweeps.py profiler=level_c 

# resume/skip existing artifacts by fixing the run timestamp label (metadata only; paths are stable)
python profiling/src/profile/cli/run_sweeps.py \
  profiler=level_a \
  'runner.components=[mmseqs2]' \
  runner.timestamp=20251124223153 \
  runner.skip_existing=true

# (zsh) quote or escape brackets to avoid globbing:
# python profiling/src/profile/cli/run_sweeps.py 'runner.components=[mmseqs2]'
```

## Useful overrides
| Purpose              | Override example                                      |
| -------------------- | ----------------------------------------------------- |
| Inspect registry     | `runner.show_components=true`                         |
| Skip re-runs         | `runner.skip_existing=true`                           |
| Container mounts     | `runner.container_runtime.mounts='["/data:/data"]'`   |
| Pin GPU              | `runner.run_config.env.CUDA_VISIBLE_DEVICES=0`        |
| Triton cache root    | `runner.run_config.env.TRITON_CACHE_DIR=/mnt/ssd/torch_extensions_cache` |
| Protenix inputs      | `runner.components=[protenix] run.sample_id=... run.quality=...` (args now flow directly to `inference_demo.sh`) |
| Cache for ESM        | `runner.run_config.env.TORCH_HOME=/mnt/...` (esm container also sets these by default) |
| Global cache envs    | Defaults now set at both `runner.run_config.env` and `runner.container_runtime.env` so every container inherits Torch/Transformers/Triton caches |
| Timestamp label      | `runner.timestamp=20240101_000000` (metadata only)    |
| Ignore Nsight errors | `runner.ignore_nsight=true`                           |
| Profiling level      | `profiler=level_c` (or `level_a`, `level_b`)          |
| Fan-out across GPUs  | `runner.run_config.gpu_ids=[0,1,2,3]` (default auto-uses visible GPUs; set `runner.run_config.concurrency=1` to pin to a single device) |

`runner.filters.scaffolds` expects short scaffold IDs (e.g., `1n0r`). Long contig strings remain in the denovo samplesheets.
Containers now inherit `runner.run_config.env`, so the Triton autotune/cache root (default `/mnt/ssd/torch_extensions_cache`) lives on the host mount instead of `/root/.triton`, avoiding recompiles across runs, and `CUDA_VISIBLE_DEVICES` falls back to the host value (or `0`) for every container.

## Smoke mode
- Use `runner.mode=smoke`; runs the same experiment combos but enforces tighter timeouts and honors filters.
- Enforces `runner.smoke.timeout_sec` per run and writes `raw/smoke/<timestamp>/smoke_report.{json,md}` under `runner.smoke.report_dir` (smoke reports still bucket by timestamp).
- Defaults to telemetry (`profiler=level_a`); override to `level_b`/`level_c` to validate Nsight quickly.

## Paths and outputs
- `profiling/configs/paths.yaml` defines inputs/outputs. `profile/layout.py` derives roots from `PROFILE_STORAGE_ROOT` (default `/mnt/nfs/new/bioinformatics/profile`).
- Run logs/summaries and profiler artifacts are grouped by level under `raw/<component>/<profiler_level>/<sample_id>/<quality>/input_<N>/output_<M>/run_<N>` (e.g., `/mnt/nfs/new/bioinformatics/profile/raw/rfdiffusion/level_a/1shg_3dx1/low/input_1/output_1/run_1`).
- Experiments live in `configs/experiments/*.yaml` (default `experiments=all`).

## Components (still declarative)
- Specs live in `configs/components/<name>.yaml` with optional helpers in `src/profile/components/<name>.py`.
- Use `${paths.outputs.foo}` / `${paths.inputs.bar}` and `${component_raw_root:foo}` for profiling artifacts (raw-only). `${component_output_root:foo}` is legacy; avoid new references.
- Optional `batching` and `apply_rules` stay minimal; each spec declares its `container` so the host can launch one container per run.

### Batch terminology (profiling sweeps)
```
python profiling/src/profile/cli/run_sweeps.py profiler=level_a experiments=minimal-batch runner.skip_existing=false
```

Note: DiffDock, AlphaFold 3, Vina-GPU, and RFdiffusion’s upstream inference script is single-sample only and sequentially processes each input.
- RFdiffusion does not support batching at all, rest of the components process inputs sequentially.
- Protenix is also forced to batch_size=1 right now due to DS4Sci EvoformerAttention asserts on batched `bias2`.

- `input_batch_size` (experiment knob) → mapped to each component’s `batching.input.key` (e.g., ProteinMPNN `batch_size`, DiffDock `batch_size`), controlling the tensor batch dimension.
- `output_samples` (experiment knob) → mapped to `batching.output.key` (e.g., ProteinMPNN `num_seq_per_target`, DiffDock `samples_per_complex`), controlling how many outputs to emit per run.
- `total_samples` is the product of the resolved input/output knobs when both exist; it shows up in run labels/paths.
- Components without a declared batch dimension ignore that knob for compute (labels still record the requested values). See the component YAML to confirm which knobs are honored.
- Intended semantics (real runs): `input_batch_size` = number of distinct inputs; `output_samples` = outputs per input. The target is `input_batch_size × output_samples` outputs, with filenames differentiating each input while sharing the same run directory. The current profiling shortcut may duplicate one placeholder input for convenience, but Nextflow will pass real distinct inputs and component scripts should name outputs per-input to avoid collisions. ProteinMPNN now follows the product rule (`batch_size * num_seq_per_target` outputs) even in the profiling shortcut.

### Minimal samples sweep
- Use `experiments=minimal-samples` to exercise only components with output fan-out knobs (rfdiffusion, diffdock, proteinmpnn, esm, protenix, alphafold3).
- Goal: confirm `output_samples` propagates and yields the requested number of outputs per input (check `.../output_<N>/run_1` under the usual profile/raw roots).
- Example: `python profiling/src/profile/cli/run_sweeps.py profiler=level_a experiments=minimal-samples runner.skip_existing=false`

## Profilers
- Strategy-based (`profiler=level_a|level_b|level_c`) under `src/profile/profilers/`; artifacts declared in configs drive `skip_existing`.
- Legacy `tools/system_utilization.py` / `tools/gpu_profile.py` are retained for reference but are no longer invoked by the runner.

## Notifications
Copy `profiling/src/profile/secret.example.yaml` → `profiling/src/profile/secret.yaml` and populate Slack/Telegram tokens. Select via `notifier=slack|telegram|none`.
