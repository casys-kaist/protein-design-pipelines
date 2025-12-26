from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[4]
PKG_ROOT = REPO_ROOT / "profiling" / "src"
for path in (PKG_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from profile.utils.denovo_locator import resolve_denovo_artifacts
from profile.utils.samples import lookup_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a DiffDock protein/ligand CSV replicated for batching.")
    parser.add_argument("--csv-path", required=True, help="Destination CSV path.")
    parser.add_argument("--protein-path", required=True, help="Protein structure path.")
    parser.add_argument("--ligand-description", required=True, help="Ligand description or path (DiffDock input).")
    parser.add_argument(
        "--protein-sequence",
        default="",
        help="Optional protein sequence column (kept blank when using structure inputs).",
    )
    parser.add_argument(
        "--complex-name",
        default="complex",
        help="Prefix for complex name entries; an index is appended for each row.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of rows to write; useful for simulating batched inputs.",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Optional sample_id to auto-fill protein_path/protein_sequence from denovo artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    count = max(1, int(args.count))

    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    if args.sample_id and (not args.protein_path or not args.protein_sequence):
        meta = lookup_sample(args.sample_id)
        dataset = meta.get("dataset")
        scaffold = meta.get("scaffold")
        ligand = meta.get("ligand")
        artifacts = resolve_denovo_artifacts(args.sample_id, scaffold=scaffold, ligand=ligand, dataset=dataset)
        if not args.protein_path:
            args.protein_path = str(artifacts.convert_pdb or artifacts.run_pdb)

    for idx in range(count):
        rows.append(
            [
                f"{args.complex_name}_{idx}",
                args.protein_path,
                args.protein_sequence,
                args.ligand_description,
            ]
        )

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["complex_name", "protein_path", "protein_sequence", "ligand_description"])
        writer.writerows(rows)

    print(f"[prep_diffdock_batch] wrote {len(rows)} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
