#!/usr/bin/env python
import argparse
import os

def merge_pdbs(pdb1_path, pdb2_path, output_pdb_path):
    """
    Merges two PDB files, assigning chain ID 'A' to the first PDB
    and chain ID 'B' to the second PDB.

    Args:
        pdb1_path (str): Path to the first PDB file (e.g., binder).
        pdb2_path (str): Path to the second PDB file (e.g., target).
        output_pdb_path (str): Path to save the merged PDB file.
    """
    if not os.path.exists(pdb1_path):
        print(f"Error: Input PDB file not found: {pdb1_path}")
        return
    if not os.path.exists(pdb2_path):
        print(f"Error: Input PDB file not found: {pdb2_path}")
        return

    with open(output_pdb_path, 'w') as outfile:
        # Process first PDB file (assign chain A)
        with open(pdb1_path, 'r') as infile1:
            for line in infile1:
                if line.startswith("ATOM  ") or line.startswith("HETATM"):
                    # Chain ID is at PDB column 22 (0-indexed 21)
                    modified_line = list(line)
                    modified_line[21] = 'A'
                    outfile.write("".join(modified_line))
                elif line.startswith("TER"):
                    # Write TER card for chain A, then can add new chain B atoms
                    # Ensure TER is correctly formatted if modifying
                    modified_ter = list(line)
                    if len(modified_ter) > 21:
                        modified_ter[21] = 'A' # Also mark chain A on TER
                    outfile.write("".join(modified_ter))
                else:
                    outfile.write(line)
        
        # Process second PDB file (assign chain B)
        with open(pdb2_path, 'r') as infile2:
            for line in infile2:
                if line.startswith("ATOM  ") or line.startswith("HETATM"):
                    modified_line = list(line)
                    modified_line[21] = 'B'
                    outfile.write("".join(modified_line))
                elif line.startswith("TER"):
                    modified_ter = list(line)
                    if len(modified_ter) > 21:
                         modified_ter[21] = 'B' # Mark chain B on TER
                    outfile.write("".join(modified_ter))
                else:
                    # Avoid writing MODEL/ENDMDL from second file if first one had them
                    if not (line.startswith("MODEL") or line.startswith("ENDMDL")):
                        outfile.write(line)

    print(f"Successfully merged PDB files into: {output_pdb_path}")
    print(f"  PDB1 ({os.path.basename(pdb1_path)}) assigned to chain A.")
    print(f"  PDB2 ({os.path.basename(pdb2_path)}) assigned to chain B.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two PDB files, assigning chain ID 'A' to the first and 'B' to the second.")
    parser.add_argument("pdb1_path", help="Path to the first PDB file (will be chain A).")
    parser.add_argument("pdb2_path", help="Path to the second PDB file (will be chain B).")
    parser.add_argument("output_pdb_path", help="Path for the merged output PDB file.")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_pdb_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    merge_pdbs(args.pdb1_path, args.pdb2_path, args.output_pdb_path)

# Example usage from command line:
# python /workspace/scripts/merge_pdbs.py path/to/binder.pdb path/to/target.pdb path/to/merged_complex.pdb 