#!/usr/bin/env python3
"""
Pick percentile ligands from CASF-2016 coreset by heavy-atom count.

Usage:
  python pick_ligand_percentiles.py <coreset_dir> \
         --pcts 15,50,85  [--list]

Notes
  • 지원 형식: *.pdb, *.pdbqt, *.mol2, *.sdf
  • H(수소)는 제외하고 heavy atom만 계산
"""
import sys, argparse, re
from pathlib import Path
from typing import Dict, List, Tuple

# ---------- Format-specific parsers ----------
def count_pdb(path: Path) -> int:
    atoms = 0
    for ln in path.read_text().splitlines():
        if ln.startswith(("HETATM", "ATOM  ")):
            elem = (ln[76:78].strip() or ln[12:16].strip()[0]).upper()
            atoms += elem != "H"
    return atoms

def count_pdbqt(path: Path) -> int:
    atoms = 0
    for ln in path.read_text().splitlines():
        if ln.startswith("ATOM"):
            elem = ln[77].upper()         # 마지막 한 글자
            atoms += elem != "H"
    return atoms

def count_mol2(path: Path) -> int:
    atoms = 0
    sec = False
    for ln in path.read_text().splitlines():
        if ln.startswith("@<TRIPOS>ATOM"):
            sec = True
            continue
        if ln.startswith("@<TRIPOS>"):
            break
        if sec:
            elem = ln.split()[-1].upper()
            atoms += elem != "H"
    return atoms

def count_sdf(path: Path) -> int:
    atoms = 0
    with path.open("rb") as fh:
        lines = fh.read().splitlines()
    if len(lines) < 4:
        return 0
    n_atoms = int(lines[3][:3])
    for ln in lines[4:4 + n_atoms]:
        elem = ln[31:34].strip().upper()
        atoms += elem != "H"
    return atoms

# ---------------- Main ----------------------
PARSERS = {
    ".pdb": count_pdb,
    ".pdbqt": count_pdbqt,
    ".mol2": count_mol2,
    ".sdf": count_sdf,
}

def collect(root: Path) -> Dict[str, int]:
    data = {}
    for d in root.iterdir():
        if not d.is_dir(): continue
        lig = None
        for ext in PARSERS:
            cand = next(d.glob(f"*ligand*{ext}"), None)
            if cand:
                lig = cand; break
        if lig is None: continue
        try:
            atoms = PARSERS[lig.suffix](lig)
            data[d.name.lower()] = atoms
        except Exception as e:
            print(f"[WARN] {d.name}: {e}", file=sys.stderr)
    if not data:
        sys.exit("No ligand files found.")
    return data

def pick(sorted_list: List[Tuple[str,int]], pct: float) -> Tuple[str,int]:
    idx = round(pct/100 * (len(sorted_list)-1))
    return sorted_list[idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coreset_dir", type=Path)
    ap.add_argument("--pcts", default="1,5,50,95,99",
                    help="comma-sep percentiles (e.g. 1,5,50,95,99)")
    ap.add_argument("--list", action="store_true",
                    help="list full table sorted by atom count")
    args = ap.parse_args()

    want = [float(p) for p in args.pcts.split(",")]
    data = collect(args.coreset_dir)
    sorted_items = sorted(data.items(), key=lambda x: x[1])

    print("#pct\tpdb_id\theavy_atoms")
    for p in want:
        pid, n = pick(sorted_items, p)
        print(f"{p}\t{pid}\t{n}")

    if args.list:
        print("\n#full_list")
        print("pdb_id\theavy_atoms")
        for pid, n in sorted_items:
            print(f"{pid}\t{n}")

if __name__ == "__main__":
    main()