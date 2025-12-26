import argparse
import os
import glob
from Bio.PDB import MMCIFParser, PDBIO

def convert_cif_to_pdb(input_dir):
    '''
    Converts all CIF (.cif) files in the specified directory to PDB (.pdb) format.
    The PDB files are saved in the same directory.

    Args:
        input_dir (str): The path to the directory containing CIF files.
    '''
    cif_files = glob.glob(os.path.join(input_dir, "*.cif"))

    if not cif_files:
        print(f'No .cif files found in directory: {input_dir}')
        return

    print(f'Found {len(cif_files)} .cif files to convert in {input_dir}')

    parser = MMCIFParser()
    io = PDBIO()

    for cif_file_path in cif_files:
        file_basename = os.path.splitext(os.path.basename(cif_file_path))[0]
        pdb_file_path = os.path.join(input_dir, f"{file_basename}.pdb")

        print(f'Converting {cif_file_path} to {pdb_file_path}...')
        try:
            # For MMCIFParser, the structure_id is often the filename base
            structure = parser.get_structure(file_basename, cif_file_path)
            io.set_structure(structure)
            io.save(pdb_file_path)
            print(f'Successfully converted and saved {pdb_file_path}')
        except Exception as e:
            print(f'Error converting {cif_file_path}: {e}')

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Convert CIF files in a directory to PDB format using Biopython.")
    # Positional argument for input directory
    arg_parser.add_argument("input_directory", help="Directory containing .cif files to convert.")
    
    args = arg_parser.parse_args()

    if not os.path.isdir(args.input_directory):
        print(f"Error: Provided path '{args.input_directory}' is not a valid directory.")
    else:
        convert_cif_to_pdb(args.input_directory) 