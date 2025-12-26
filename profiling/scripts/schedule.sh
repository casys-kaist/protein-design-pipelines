#!/bin/bash
# Schedule a couple of profiling sweeps.
# Run from profiling/ with: bash scripts/schedule.sh

python src/profile/cli/run_sweeps.py \
  runner.components='["vina_gpu"]' \
  runner.repeats=1 \
  runner.list_runs=true

python src/profile/cli/run_sweeps.py \
  runner.components='["proteinmpnn"]' \
  runner.filters.scaffolds='["1n0r","1tim"]' \
  runner.repeats=2
