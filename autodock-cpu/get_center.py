#
# Corrected Protein-Ligand Docking Script using Vina-Python
#
import argparse
import numpy as np
import os
from pathlib import Path
from vina import Vina
# openbabel is required for the conversion function, if you use it.
from openbabel import pybel

def get_structure_center(pdbqt_path: str) -> list[float]:
    """
    Return the geometric center of a PDBQT structure (receptor or ligand).
    """
    coords = []
    try:
        with open(pdbqt_path, "r") as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
        
        if not coords:
            raise ValueError(f"No ATOM or HETATM records found in {pdbqt_path}")

        arr = np.array(coords)
        return arr.mean(axis=0).tolist()
    except (IOError, ValueError) as e:
        print(f"Error processing file {pdbqt_path}: {e}")
        raise



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the center of a PDBQT structure."
    )
    parser.add_argument(
        "--receptor_pdbqt_path",
        type=str,
        required=True,
        help="Path of the receptor PDBQT file.",
    )

    args = parser.parse_args()
    center = get_structure_center(args.receptor_pdbqt_path)
    print(f"center_x={center[0]:.3f}")
    print(f"center_y={center[1]:.3f}")
    print(f"center_z={center[2]:.3f}")