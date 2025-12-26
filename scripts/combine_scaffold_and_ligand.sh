#!/usr/bin/env bash
# Combine a scaffold PDB with a docked ligand PDBQT into a single PDB.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  combine_scaffold_and_ligand.sh --base-dir <dir> --scaffold <scaffold_id> --ligand <ligand_id> --output <name> [--docked-pdbqt <file>] [--docked-dir <dir>]

Example:
  combine_scaffold_and_ligand.sh \
    --base-dir /mnt/nfs/new/bioinformatics/handpicked \
    --scaffold 1shg \
    --ligand 3dx1 \
    --output 1shg_3dx1

Defaults:
  --docked-dir <base-dir>/nextflow (searched recursively for "*_docked.pdbqt")
  output path: <base-dir>/docked/<output>.pdb
EOF
}

BASE_DIR=""
SCAFFOLD=""
LIGAND=""
OUTPUT_NAME=""
DOCKED_PDBQT=""
DOCKED_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-dir) BASE_DIR="$2"; shift 2 ;;
    --scaffold) SCAFFOLD="$2"; shift 2 ;;
    --ligand) OUTPUT_LIGAND="$2"; LIGAND="$2"; shift 2 ;;
    --output) OUTPUT_NAME="$2"; shift 2 ;;
    --docked-pdbqt) DOCKED_PDBQT="$2"; shift 2 ;;
    --docked-dir) DOCKED_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$BASE_DIR" || -z "$SCAFFOLD" || -z "$LIGAND" || -z "$OUTPUT_NAME" ]]; then
  echo "Missing required args"; usage; exit 1
fi

BASE_DIR=$(realpath "$BASE_DIR")
SCAFFOLD_PDB="${BASE_DIR}/scaffolds/${SCAFFOLD}.pdb"
OUT_DIR="${BASE_DIR}/docked"
mkdir -p "$OUT_DIR"

if [[ -z "$DOCKED_DIR" ]]; then
  DOCKED_DIR="${BASE_DIR}/nextflow"
fi

if [[ -n "$DOCKED_PDBQT" ]]; then
  DOCKED_PDBQT=$(realpath "$DOCKED_PDBQT")
else
  mapfile -t candidates < <(find "$DOCKED_DIR" -type f \( -name "*${LIGAND}*_docked.pdbqt" -o -name "*${SCAFFOLD}*_docked.pdbqt" \) | sort)
  if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "No docked pdbqt found under ${DOCKED_DIR} for ligand ${LIGAND} or scaffold ${SCAFFOLD}" >&2
    exit 1
  fi
  DOCKED_PDBQT="${candidates[0]}"
fi

for tool in pdb_chain pdb_rplresname obabel; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing required tool: $tool" >&2
    exit 1
  fi
done

if [[ ! -f "$SCAFFOLD_PDB" ]]; then
  echo "Scaffold PDB not found: $SCAFFOLD_PDB" >&2
  exit 1
fi

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

scaffold_chainA="${tmpdir}/scaffold_chainA.pdb"
ligand_pdb="${tmpdir}/ligand.pdb"
ligand_chainB="${tmpdir}/ligand_chainB.pdb"

# Chain A: scaffold without END
pdb_chain -A "$SCAFFOLD_PDB" | grep -v '^END' > "$scaffold_chainA"

# Ligand: pdbqt -> pdb (first pose only, drop MODEL/END lines)
obabel "$DOCKED_PDBQT" -o pdb -d -f 1 -l 1 | grep -v -E '^(MODEL|END)' > "$ligand_pdb"

# Chain B: rename ATOM to HETATM, set chain B, rename UNL/MOL -> LIG
pdb_chain -B "$ligand_pdb" \
  | awk '{if ($0 ~ /^ATOM  /) sub(/^ATOM  /,"HETATM"); print}' \
  | pdb_rplresname -UNK:LIG \
  | pdb_rplresname -UNL:LIG \
  | pdb_rplresname -MOL:LIG \
  > "$ligand_chainB"

output_pdb="${OUT_DIR}/${OUTPUT_NAME}.pdb"
{
  cat "$scaffold_chainA"
  echo 'TER'
  cat "$ligand_chainB"
  echo 'TER'
  echo 'END'
} > "$output_pdb"

echo "Wrote combined PDB: $output_pdb"
