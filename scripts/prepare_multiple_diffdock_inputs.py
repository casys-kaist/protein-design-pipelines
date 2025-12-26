import argparse
import csv
import os
import glob

# Define the example ligand (SMILES string from DiffDock\'s example)
EXAMPLE_LIGAND_SMILES = "COc(cc1)ccc1C#N"

def batch_create_diffdock_input_csv(protein_base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for subdir_name in os.listdir(protein_base_dir):
        subdir_path = os.path.join(protein_base_dir, subdir_name)

        if not os.path.isdir(subdir_path):
            continue
        
        output_csv_path = os.path.join(output_dir, f"{subdir_name}.csv")

        create_diffdock_input_csv(subdir_path, output_csv_path)

def create_diffdock_input_csv(protein_dir, output_csv_path):
    """
    Generates a CSV file for DiffDock input.

    Args:
        protein_dir (str): Path to the directory containing protein PDB files.
        output_csv_path (str): Path to save the generated CSV file.
    """
    protein_pdb_files = glob.glob(os.path.join(protein_dir, "*.pdb"))
    
    if not protein_pdb_files:
        print(f"No PDB files found in {protein_dir}. Please ensure PDB files are present.")
        print("If you have CIF files, please convert them to PDB format first.")
        return

    print(f"Found {len(protein_pdb_files)} PDB files in {protein_dir}.")

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_csv_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['complex_name', 'protein_path', 'ligand_description', 'protein_sequence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, pdb_file_path in enumerate(protein_pdb_files):
            protein_basename = os.path.splitext(os.path.basename(pdb_file_path))[0]
            complex_name = f"{protein_basename}_exampleligand_{i+1}"
            
            # IMPORTANT: The protein_path written to the CSV must be the path
            # ACCESSIBLE FROM WITHIN THE DIFFDOCK DOCKER CONTAINER.
            # This script assumes that the input protein_dir is already a path
            # that will be valid inside the container (e.g., on a mounted volume).
            # If protein_dir is a host path that\'s mapped differently into the
            # container, this path will need adjustment before writing to the CSV,
            # or the CSV needs to be generated/edited from within the container
            # or with container-aware paths.

            writer.writerow({
                'complex_name': complex_name,
                'protein_path': pdb_file_path, # This path must be valid inside the Docker container
                'ligand_description': EXAMPLE_LIGAND_SMILES,
                'protein_sequence': '' # Assuming PDB is provided, so sequence is not needed
            })
    print(f"Successfully generated DiffDock input CSV: {output_csv_path}")
    print(f"It lists {len(protein_pdb_files)} complexes with the example ligand: {EXAMPLE_LIGAND_SMILES}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DiffDock input CSV from protein PDB files and an example ligand.")
    parser.add_argument("--protein_dir", required=True, help="Directory containing directories of protein PDB files. These paths must be accessible from within the DiffDock container.")
    parser.add_argument("--output_dir", default="diffdock_inputs", help="Directory to save the generated DiffDock input CSV files. This path should also be accessible from the container.")
    
    args = parser.parse_args()
    batch_create_diffdock_input_csv(args.protein_dir, args.output_dir)

'''
    print("\\nReminder:")
    print("1. Ensure the paths in the generated CSV (especially 'protein_path') are accessible from *inside* the DiffDock Docker container.")
    print("   For example, if your PDBs are in /home/user/data/proteins on the host and this is mounted to /mnt/data/proteins in the container,")
    print("   the script should be run with --protein_dir /mnt/data/proteins (or paths adjusted accordingly).")
    print("2. If your protein files are in CIF format, convert them to PDB before running this script.")
    print("3. Ensure the --output_csv path is also accessible from within the container when you run DiffDock.") 
'''