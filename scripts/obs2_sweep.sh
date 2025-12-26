#!/usr/bin/env bash
set -euo pipefail

# Obs#2 fan-out sweep helper (K8s-first)
# Levels: 1|2|3 or 'all' to run 1,2,3 sequentially.
# Fan-out knobs (uniform per level):
#  - RFDiffusion: --rfdiffusion_num_designs
#  - ProteinMPNN: --proteinmpnn_num_seq_per_target
#  - ESM:         --esm_num_variants
#  - Protenix:    --protenix_num_samples

LEVELS=("1" "2" "3")
# Default: a ready-to-run denovo sample (has contigs)
INPUT_SHEET="nextflow/samplesheet/denovo/denovo_1shg_3dx1.csv"
# Default profile; allow override via env PROFILE or --profile flag
PROFILE=${PROFILE:-"k8s"}

usage() {
  echo "Usage: $0 <1|2|3|all> [--profile <k8s|docker|...>] [--input <samplesheet.csv>] [--outdir <dir>] [--extra '<extra nextflow args>']" >&2
}

LEVEL_ARG=${1:-}
if [[ -z "${LEVEL_ARG}" ]]; then
  usage; exit 1
fi
shift || true

OUTDIR_BASE=${OUTDIR_BASE:-"results/obs2"}
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2;;
    --input) INPUT_SHEET="$2"; shift 2;;
    --outdir) OUTDIR_BASE="$2"; shift 2;;
    --extra) EXTRA_ARGS="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ "${LEVEL_ARG}" != "all" ]]; then
  LEVELS=("${LEVEL_ARG}")
fi

timestamp() { date +%Y%m%d-%H%M%S; }

for LVL in "${LEVELS[@]}"; do
  case "${LVL}" in
    1|2|3) :;;
    *) echo "Invalid level: ${LVL}. Expected 1,2,3." >&2; exit 3;;
  esac

  OUTDIR="${OUTDIR_BASE}/fanout_${LVL}_$(timestamp)"
  echo "[Obs#2] Running level ${LVL} → ${OUTDIR}"

  nextflow run nextflow/main.nf \
    -w nextflow/work \
    -profile "${PROFILE}" \
    --pipeline denovo_design_ligand \
    --input "${INPUT_SHEET}" \
    --structure_design_mode rfdiffusion \
    --inverse_folding_mode proteinmpnn \
    --variant_generation_mode esm \
    --msa_mode mmseqs2 \
    --structure_prediction_mode protenix \
    --outdir "${OUTDIR}" \
    --rfdiffusion_num_designs "${LVL}" \
    --proteinmpnn_num_seq_per_target "${LVL}" \
    --esm_num_variants "${LVL}" \
    --protenix_num_samples "${LVL}" \
    ${EXTRA_ARGS}
done

echo "[Obs#2] Sweep complete. Outputs under: ${OUTDIR_BASE}"
