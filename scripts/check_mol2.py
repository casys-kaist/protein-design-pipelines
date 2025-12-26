#!/usr/bin/env python3
"""
Quick mol2 sanity checker.

Checks:
- parseable ATOM/BOND sections
- atom/bond counts
- heavy-atom count
- coordinate spread (flags collapsed/all-zero coordinates)

Exit code is non-zero if any file fails to parse or has zero atoms.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


def _iter_mol2_files(targets: Iterable[str], recursive: bool) -> Iterable[Path]:
    for raw in targets:
        path = Path(raw)
        if path.is_dir():
            pattern = "**/*.mol2" if recursive else "*.mol2"
            yield from path.glob(pattern)
        elif path.suffix.lower() == ".mol2":
            yield path


def _infer_element(name: str, atom_type: str) -> str:
    def _norm(token: str) -> str:
        letters = "".join(ch for ch in token if ch.isalpha())
        if not letters:
            return ""
        if len(letters) >= 2 and letters[1].islower():
            return letters[:2].capitalize()
        return letters[0].upper()

    for token in (atom_type, name):
        elem = _norm(token)
        if elem:
            return elem
    return ""


def _parse_mol2(path: Path) -> Tuple[List[Tuple[str, str, Tuple[float, float, float]]], List[Tuple[int, int]]]:
    atoms: List[Tuple[str, str, Tuple[float, float, float]]] = []
    bonds: List[Tuple[int, int]] = []
    in_atoms = False
    in_bonds = False

    for line in path.read_text().splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms, in_bonds = True, False
            continue
        if line.startswith("@<TRIPOS>BOND"):
            in_atoms, in_bonds = False, True
            continue
        if line.startswith("@<TRIPOS>"):
            in_atoms, in_bonds = False, False
            continue

        if in_atoms:
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Malformed ATOM line: {line}")
            name = parts[1]
            try:
                x, y, z = (float(parts[2]), float(parts[3]), float(parts[4]))
            except ValueError as exc:
                raise ValueError(f"Non-numeric coordinates in line: {line}") from exc
            atom_type = parts[5]
            atoms.append((name, atom_type, (x, y, z)))
        elif in_bonds:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                a, b = int(parts[1]), int(parts[2])
            except ValueError:
                continue
            bonds.append((a, b))

    return atoms, bonds


def _span(coords: List[Tuple[float, float, float]]) -> float:
    xs, ys, zs = zip(*coords)
    return math.sqrt(
        (max(xs) - min(xs)) ** 2
        + (max(ys) - min(ys)) ** 2
        + (max(zs) - min(zs)) ** 2
    )


def check_file(path: Path) -> Tuple[str, List[str], int, int, int, float]:
    issues: List[str] = []
    status = "OK"
    try:
        atoms, bonds = _parse_mol2(path)
    except Exception as exc:  # pragma: no cover - defensive
        return "FAIL", [f"parse error: {exc}"], 0, 0, 0, 0.0

    atom_count = len(atoms)
    bond_count = len(bonds)
    heavy_atoms = sum(1 for name, atom_type, _ in atoms if _infer_element(name, atom_type) != "H")

    if atom_count == 0:
        status = "FAIL"
        issues.append("no atoms")
    if bond_count == 0:
        issues.append("no bonds")
    if heavy_atoms <= 1:
        issues.append("<=1 heavy atom")

    coords = [c for _, _, c in atoms]
    span = _span(coords) if coords else 0.0
    if coords and not all(math.isfinite(v) for xyz in coords for v in xyz):
        status = "FAIL"
        issues.append("non-finite coords")
    if coords and span < 1e-3:
        issues.append("collapsed coordinates (zero span)")

    if status != "FAIL" and issues:
        status = "WARN"
    return status, issues, atom_count, heavy_atoms, bond_count, span


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity-check mol2 files.")
    parser.add_argument("paths", nargs="+", help="mol2 files or directories to scan")
    parser.add_argument("--recursive", "-r", action="store_true", help="recurse into directories")
    args = parser.parse_args()

    files = list(_iter_mol2_files(args.paths, args.recursive))
    if not files:
        print("No mol2 files found.", file=sys.stderr)
        return 1

    worst_status = "OK"
    print("status\tatoms\theavy\tbonds\tspan_A\tpath\tissues")
    for path in sorted(files):
        status, issues, atoms, heavy_atoms, bonds, span = check_file(path)
        if status == "FAIL":
            worst_status = "FAIL"
        elif status == "WARN" and worst_status == "OK":
            worst_status = "WARN"
        issue_text = "; ".join(issues)
        print(f"{status}\t{atoms}\t{heavy_atoms}\t{bonds}\t{span:.2f}\t{path}\t{issue_text}")

    return 1 if worst_status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
