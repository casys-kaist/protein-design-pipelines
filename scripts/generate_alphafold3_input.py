#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def read_fasta_sequences(fasta_file):
    """
    Read all sequences from a FASTA file.
    Return list of tuples: [(header, sequence), ...]
    """
    records = []
    header = None
    seq_chunks = []
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                # flush previous
                if header is not None:
                    records.append((header, ''.join(seq_chunks)))
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if header is not None:
        records.append((header, ''.join(seq_chunks)))
    return records


def create_msa_files(msa_file: str, msa_dir: str, query_sequence: str) -> None:
    """Create pairing.a3m and non_pairing.a3m files from the input MSA file."""
    with open(msa_file, "r") as f:
        lines = f.readlines()

    with open(os.path.join(msa_dir, "pairing.a3m"), 'w') as f:
        indexed_sequence = "".join(query_sequence)
        f.write(f">UniRef100_query_0/\n{indexed_sequence}\n")

    with open(os.path.join(msa_dir, "non_pairing.a3m"), 'w') as f:
        f.write(f">query\n{query_sequence}\n")
        current_header = None
        current_seq = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_header and current_seq:
                    f.write(f"{current_header}\n{''.join(current_seq)}\n")
                current_header = line
                current_seq = []
            else:
                current_seq.append(line)
        if current_header and current_seq:
            f.write(f"{current_header}\n{''.join(current_seq)}\n")


def _build_fold_job(query_sequence: str, base_name: str, num_samples: int, seed_offset: int) -> dict:
    # Keep a single seed per input; output_samples controls diffusion samples.
    seeds = [seed_offset + 1]
    return {
        "name": base_name,
        "sequences": [
            {
                "protein": {
                    "id": ["A"],
                    "sequence": query_sequence,
                    "unpairedMsaPath": "./pairing.a3m",
                    "pairedMsaPath": "./non_pairing.a3m",
                    "templates": [],
                }
            }
        ],
        "modelSeeds": seeds,
        "dialect": "alphafold3",
        "version": 1,
    }


def generate_alphafold3_json(
    msa_file: str,
    fasta_file: str,
    output_dir: str,
    num_samples: int,
    *,
    num_inputs: int = 1,
    name_prefix: str | None = None,
) -> list[dict]:
    """
    Generate AlphaFold3 input JSON(s) and required A3M files into the given output directory.

    If FASTA contains multiple sequences, a warning is emitted and only the first is used.
    """
    msa_path = Path(msa_file).expanduser().resolve()
    fasta_path = Path(fasta_file).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not msa_path.exists():
        raise FileNotFoundError(f"MSA file not found: {msa_path}")
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1 (got {num_samples})")
    if num_inputs < 1:
        raise ValueError(f"num_inputs must be >= 1 (got {num_inputs})")

    print(f"Using MSA file: {msa_path}")
    print(f"Using FASTA file: {fasta_path}")
    print(f"Output directory: {out_dir}")
    print(f"Requested inputs: {num_inputs}, samples per input: {num_samples}")

    records = read_fasta_sequences(str(fasta_path))
    if len(records) == 0:
        raise ValueError("Could not read sequence from FASTA file")
    if len(records) > 1:
        first_header = records[0][0] or "seq0"
        print(
            f"[WARN] Multiple sequences detected in FASTA ({len(records)}). "
            f"Using only the first one: {first_header}"
        )

    query_sequence = records[0][1]
    if not query_sequence:
        raise ValueError("First sequence is empty in FASTA file")

    create_msa_files(str(msa_path), str(out_dir), query_sequence)

    base_name = name_prefix or Path(msa_path).stem
    json_payloads: list[dict] = []

    for idx in range(num_inputs):
        job_name = base_name if num_inputs == 1 else f"{base_name}_b{idx + 1}"
        seed_offset = idx * num_samples
        payload = _build_fold_job(query_sequence, job_name, num_samples, seed_offset)
        filename = "alphafold3_input.json" if idx == 0 else f"alphafold3_input_{idx + 1}.json"
        json_path = out_dir / filename
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=4)
        json_payloads.append({"name": job_name, "path": str(json_path), "seeds": payload["modelSeeds"]})
        print(f"Wrote {json_path} with seeds {payload['modelSeeds']}")

    return json_payloads

def main():
    parser = argparse.ArgumentParser(
        description="Generate AlphaFold3 input JSON and A3M files into an output directory"
    )
    parser.add_argument("msa_file", help="Input MSA in A3M format")
    parser.add_argument("--fasta", "-f", required=True, help="Input FASTA file")
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory to write alphafold3_input.json and A3M files",
    )
    parser.add_argument(
        "--num_samples",
        "-n",
        required=True,
        type=int,
        help="Number of diffusion samples per input (seeds stay at 1)",
    )
    parser.add_argument(
        "--count",
        "-c",
        type=int,
        default=1,
        help="How many fold inputs to emit (controls batch size)",
    )
    parser.add_argument(
        "--name-prefix",
        help="Optional base name for the generated jobs (defaults to MSA filename stem)",
    )
    args = parser.parse_args()

    try:
        generate_alphafold3_json(
            args.msa_file,
            args.fasta,
            args.output,
            int(args.num_samples),
            num_inputs=int(args.count),
            name_prefix=args.name_prefix,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
