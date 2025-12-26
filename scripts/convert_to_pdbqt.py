#!/usr/bin/env python3
"""
Convert PDB, SDF, or SMILES into AutoDock-Vina-compatible PDBQT.

⋅ ligand  mode : keeps ROOT/BRANCH torsion-tree tags, adds Gasteiger charges
⋅ receptor mode: omits torsion-tree tags for a rigid target

Dependencies: openbabel-3.x, pybel
$ conda install -c conda-forge openbabel
"""

import argparse
import sys
from pathlib import Path
from openbabel import openbabel as ob
from openbabel import pybel

TORSION_TAGS = {"ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF"}


def _calc_gasteiger(mol: ob.OBMol) -> None:
    """Assign Gasteiger partial charges in place."""
    model = ob.OBChargeModel.FindType("gasteiger")
    if not (model and model.ComputeCharges(mol)):
        sys.exit("Gasteiger charge calculation failed (Open Babel).")


def _strip_torsion_tree(path: Path) -> None:
    """Remove torsion-tree lines (ROOT/BRANCH…) if any remain."""
    with path.open() as fh:
        lines = [
            ln
            for ln in fh
            if not any(ln.lstrip().startswith(tag) for tag in TORSION_TAGS)
        ]
    path.write_text("".join(lines))


def _write_pdbqt(
    mol: pybel.Molecule,
    path: Path,
    mode: str,
) -> None:
    """Write PDBQT using OBConversion to control torsion-tree options."""
    conv = ob.OBConversion()
    conv.SetOutFormat("pdbqt")

    if mode == "receptor":
        # 'r' flag → rigid receptor (no ROOT/BRANCH)
        conv.AddOption("r", conv.OUTOPTIONS)
        # 'A' flag → write all atoms including hydrogens
        conv.AddOption("A", conv.OUTOPTIONS)
    else:  # ligand
        # keep ROOT/BRANCH, ensure charges present
        conv.AddOption("A", conv.OUTOPTIONS)  # all atoms
        conv.AddOption("g", conv.OUTOPTIONS)  # write Gasteiger charges

    conv.WriteFile(mol.OBMol, str(path))
    conv.CloseOutFile()

    # extra safety: strip any residual tags in receptor file
    if mode == "receptor":
        _strip_torsion_tree(path)


def convert(in_file: Path, out_file: Path, mode: str) -> None:
    if not in_file.exists():
        sys.exit(f"Input file not found: {in_file}")

    # --- Detect input format: PDB, SDF, or SMILES ---
    ext = in_file.suffix.lower()
    if ext == ".sdf":
        mol = next(pybel.readfile("sdf", str(in_file)))
    elif ext in {".pdb", ".ent"}:
        mol = next(pybel.readfile("pdb", str(in_file)))
    else:  # assume SMILES (.smi or plain text)
        mol = next(pybel.readfile("smi", str(in_file)))

    if not mol:
        sys.exit("Failed to read the molecule.")

    # --- Add hydrogens (and for ligands: 3-D + charges) ---
    mol.addh()
    if mode == "ligand":
        mol.make3D(forcefield="mmff94", steps=500)
        _calc_gasteiger(mol.OBMol)

    _write_pdbqt(mol, out_file, mode)
    print(f"{in_file.name} → {out_file.name} ({mode})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDB/SDF/SMILES to PDBQT (ligand or receptor)."
    )
    parser.add_argument("--in_file", required=True, help="Input molecule file")
    parser.add_argument("--out_pdbqt", required=True, help="Output PDBQT file")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["ligand", "receptor"],
        help="'ligand' keeps torsion tree; 'receptor' omits it.",
    )
    args = parser.parse_args()
    convert(Path(args.in_file), Path(args.out_pdbqt), args.mode)


if __name__ == "__main__":
    main()