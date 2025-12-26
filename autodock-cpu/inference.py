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

def run_inference(
    protein_path: str, 
    ligand_path: str, 
    output_path: str, 
    center: list[float],
    box_size: list[float], 
    exhaustiveness: int, 
    n_poses: int
) -> None:
    """
    Dock the ligand into the protein using AutoDock Vina.
    The search space is centered on the geometric center of the receptor.
    """
    v = Vina(sf_name="vina")
    
    print(f"Setting receptor from: {protein_path}")
    v.set_receptor(protein_path)
    
    print(f"Setting ligand from: {ligand_path}")
    v.set_ligand_from_file(ligand_path)
    
    # --- CRITICAL FIX ---
    # The center of the search box is now calculated from the RECEPTOR.
    print(f"Computed Receptor Center: x={center[0]:.3f}, y={center[1]:.3f}, z={center[2]:.3f}")
    print(f"Using Box Size: x={box_size[0]}, y={box_size[1]}, z={box_size[2]}")
    
    v.compute_vina_maps(center=center, box_size=box_size)
    
    # The 'dock' method performs the entire docking process,
    # including scoring and optimization of poses.
    print(f"Performing docking with exhaustiveness={exhaustiveness}...")
    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    
    # Writes the docked poses to a PDBQT file.
    v.write_poses(output_path, n_poses=n_poses, overwrite=True)
    print("Docking complete.")


def main(args):
    """
    Main function to set up paths and run the docking inference.
    """
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a more descriptive output file name
    ligand_name = Path(args.ligand_pdbqt_path).stem
    receptor_name = Path(args.input_pdbqt_path).stem
    output_file_name = f"{receptor_name}_{ligand_name}_docked.pdbqt"
    output_docking_result = results_dir / output_file_name

    center = [args.center_x, args.center_y, args.center_z]
    box_size = [args.size_x, args.size_y, args.size_z]
    
    try:
        run_inference(
            protein_path=args.input_pdbqt_path, 
            ligand_path=args.ligand_pdbqt_path, 
            output_path=str(output_docking_result),
            center=center,
            box_size=box_size,
            exhaustiveness=args.exhaustiveness,
            n_poses=args.n_poses
        )
        print(f"\nSuccessfully saved docking result to: {output_docking_result}")
    except Exception as e:
        print(f"\nAn error occurred during the docking process: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Protein-ligand docking with Vina. The search box is automatically centered on the receptor."
    )
    parser.add_argument(
        "--input_pdbqt_path",
        type=str,
        required=True,
        help="Path of the receptor PDBQT file.",
    )
    parser.add_argument(
        "--ligand_pdbqt_path",
        type=str,
        required=True,
        help="Path of the ligand PDBQT file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory where the output files will be written.",
    )
    parser.add_argument(
        "--center_x", type=float, required=True, help="X coordinate of the center of the docking box."
    )
    parser.add_argument(
        "--center_y", type=float, required=True, help="Y coordinate of the center of the docking box."
    )
    parser.add_argument(
        "--center_z", type=float, required=True, help="Z coordinate of the center of the docking box."
    )
    parser.add_argument(
        "--size_x", type=float, default=25.0, help="Size of the docking box in X dimension (Angstrom)."
    )
    parser.add_argument(
        "--size_y", type=float, default=25.0, help="Size of the docking box in Y dimension (Angstrom)."
    )
    parser.add_argument(
        "--size_z", type=float, default=25.0, help="Size of the docking box in Z dimension (Angstrom)."
    )
    parser.add_argument(
        "--exhaustiveness", type=int, default=32, help="Exhaustiveness of the global search."
    )
    parser.add_argument(
        "--n_poses", type=int, default=20, help="Number of binding modes to generate."
    )
    
    args = parser.parse_args()
    main(args)