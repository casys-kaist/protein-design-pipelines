# ESM-based Protein Sequence Variant Generation

This directory contains a script for generating protein sequence variants using ESM (Evolutionary Scale Modeling) language models.

## `generate_variants.py`

This script uses a specified ESM model (e.g., ESM-2) to generate variants of a given reference protein sequence. It supports both generating single-mutation variants and iteratively generating multi-mutation variants.

### Functionality

1.  **Input**: Takes a reference protein sequence either directly as a string (`--ref_sequence`) or from the first sequence in a FASTA file (`--ref_fasta_file`).
2.  **Model Loading**: Loads a specified Hugging Face ESM model (e.g., `facebook/esm2_t33_650M_UR50D`). It utilizes the `accelerate` library for automatic device mapping (`device_map="auto"`), allowing large models to be distributed across available hardware (GPUs/CPU) to manage memory usage.
3.  **Variant Generation**: 
    *   **Iterative Mutations**: If `--max_mutations` is set to `N > 1`, the script generates variants by iteratively applying mutations. It starts with the reference, predicts and applies one mutation, then uses the resulting sequence to predict and apply the next mutation, up to `N` times. It aims to generate a total of `--num_variants`, distributing them across mutation counts from 1 to `N`.
    *   **Single Mutations**: If `--max_mutations` is 1 (the default), the script generates `--num_variants` unique single-mutation variants by masking and predicting at different positions in the original reference sequence.
    *   **Prediction Method**: At each mutation step (or for each single mutant), it masks a randomly chosen, not-yet-mutated position. The ESM model predicts likely amino acid substitutions. 
    *   **Sampling**: It samples a new residue from the `top_k` most probable *valid* amino acid predictions (excluding the original residue at that position), weighted by their predicted probabilities.
4.  **Output**: Saves the generated variants, along with the reference sequence and mutation details (position, original/new residue, step-by-step history for iterative mutations), into a single JSON file specified by `--output_file`.

### Usage

```bash
cd /workspace/esm

# Example: Generate 10 single-mutation variants for a given sequence
python generate_variants.py \
    --ref_sequence "MAALEKLN..." \
    --model_name "facebook/esm2_t33_650M_UR50D" \
    --num_variants 10 \
    --output_file "results/my_protein_single_variants.json" \
    --top_k 5 \
    --cache_dir /path/to/hf/cache # Optional: Specify cache directory

# Example: Generate 20 variants with up to 3 iterative mutations using a FASTA input
python generate_variants.py \
    --ref_fasta_file "../inputs/my_protein.fasta" \
    --model_name "facebook/esm2_t33_650M_UR50D" \
    --num_variants 20 \
    --max_mutations 3 \
    --output_file "results/my_protein_multi_variants.json" \
    --top_k 5
```

### Key Arguments

*   `--ref_sequence`: Provide the reference sequence directly.
*   `--ref_fasta_file`: Provide the path to a FASTA file containing the reference sequence.
*   `--output_file`: Path to save the output JSON file (default: `results/esm_variants.json`).
*   `--num_variants`: Total number of variants to generate (default: 10).
*   `--max_mutations`: Maximum number of iterative mutations per variant. `1` generates only single mutants (default: 1).
*   `--model_name`: Hugging Face model name (default: `facebook/esm2_t33_650M_UR50D`).
*   `--top_k`: Sample mutations from the top K predictions (default: 5).
*   `--cache_dir`: Optional Hugging Face cache directory.
*   `--skip_existing`: If set, skip generation if the output file already exists.

**Note**: Requires `torch`, `transformers`, `accelerate`, and `biopython` (`pip install torch transformers accelerate biopython`).

## Dependencies

*   Python 3.x
*   PyTorch (`torch`)
*   Hugging Face Transformers (`transformers`)
*   Biopython (`biopython`) - Optional, required only for FASTA input support.

Install dependencies using pip:
```bash
pip install torch transformers biopython
```

## Future Work / Ideas

*   **Advanced Selection:** Enhance `select_sequence.py` or create new scripts to filter/rank variants based on model confidence scores, predicted stability, or other metrics.
*   **Targeted Mutagenesis:** Integrate structural or functional information to guide mutations towards specific regions of interest.
*   **Alternative Iteration Strategies:** Explore different ways to select the base sequence for the next mutation step (e.g., based on model likelihood). 
