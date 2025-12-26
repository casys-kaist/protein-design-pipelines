import argparse
import json
import torch
import random
import logging
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pathlib import Path
import math
from collections import defaultdict
import sys
try:
    from Bio import SeqIO # Needs biopython: pip install biopython
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    # Don't print warning unless user tries to use FASTA input

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Function for Single Mutation Step ---
def perform_one_mutation(current_sequence, model, tokenizer, top_k, input_device, mutated_positions, batch_size=1):
    """
    Attempts to introduce a single mutation into the sequence at a position
    that has not been mutated before in the current iterative chain.
    Evaluates up to `batch_size` candidate positions per forward pass.
    Returns (new_sequence, position, original_residue, new_residue) or None if failed.
    """
    seq_len = len(current_sequence)
    possible_indices = list(set(range(seq_len)) - mutated_positions)
    max_attempts_per_step = min(len(possible_indices) * 3, 30)
    attempts = 0
    batch_size = max(1, int(batch_size)) if batch_size else 1

    while attempts < max_attempts_per_step and possible_indices:
        attempts += 1
        batch_indices = random.sample(possible_indices, k=min(batch_size, len(possible_indices)))

        masked_sequences = []
        original_residues = []
        for mut_idx in batch_indices:
            original_residue = current_sequence[mut_idx]
            masked_sequence = list(current_sequence)
            masked_sequence[mut_idx] = tokenizer.mask_token
            masked_sequences.append("".join(masked_sequence))
            original_residues.append(original_residue)

        try:
            inputs = tokenizer(masked_sequences, return_tensors="pt", padding=True).to(input_device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits.to(input_device)

            mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=False)
            mask_index_map = {}
            for row_idx, col_idx in mask_positions.tolist():
                if row_idx not in mask_index_map:
                    mask_index_map[row_idx] = col_idx

            for idx_in_batch, mut_idx in enumerate(batch_indices):
                mask_token_index = mask_index_map.get(idx_in_batch)
                if mask_token_index is None:
                    possible_indices.remove(mut_idx)
                    continue

                masked_token_logits = logits[idx_in_batch, mask_token_index, :]
                probabilities = torch.softmax(masked_token_logits, dim=-1)
                top_k_probs, top_k_indices = torch.topk(probabilities, top_k, dim=-1)

                top_k_tokens = [tokenizer.decode(idx.item()) for idx in top_k_indices.cpu()]
                valid_indices = [
                    i for i, token in enumerate(top_k_tokens)
                    if len(token.strip()) == 1 and token.strip().isupper() and token.strip() in "ACDEFGHIKLMNPQRSTVWY"
                ]

                if not valid_indices:
                    possible_indices.remove(mut_idx)
                    continue

                filtered_probs = top_k_probs.cpu()[valid_indices]
                filtered_tokens = [top_k_tokens[i] for i in valid_indices]

                prob_sum = torch.sum(filtered_probs)
                if prob_sum <= 0:
                    possible_indices.remove(mut_idx)
                    continue
                normalized_probs = filtered_probs / prob_sum

                sample_attempts = 0
                max_sample_attempts = min(len(set(filtered_tokens) - {original_residues[idx_in_batch]}), 5)
                while sample_attempts < max_sample_attempts:
                    sampled_token_index = torch.multinomial(normalized_probs, 1).item()
                    new_residue = filtered_tokens[sampled_token_index]
                    if new_residue != original_residues[idx_in_batch]:
                        variant_sequence_list = list(current_sequence)
                        variant_sequence_list[mut_idx] = new_residue
                        new_sequence = "".join(variant_sequence_list)
                        return new_sequence, mut_idx, original_residues[idx_in_batch], new_residue
                    sample_attempts += 1

                possible_indices.remove(mut_idx)

        except RuntimeError as e:
            if "out of memory" in str(e).lower() and (input_device.type == 'cuda'):
                logging.error(f"CUDA out of memory error during mutation prediction at batched positions {[i + 1 for i in batch_indices]}: {e}. Try a smaller model or ensure sufficient GPU/CPU RAM.")
                raise e
            else:
                logging.warning(f"Runtime error during prediction for positions {[i + 1 for i in batch_indices]}: {e}. Skipping batch.")
                for mut_idx in batch_indices:
                    if mut_idx in possible_indices:
                        possible_indices.remove(mut_idx)
                continue
        except Exception as e:
            logging.warning(f"Unexpected error during prediction for positions {[i + 1 for i in batch_indices]}: {e}. Skipping batch.")
            for mut_idx in batch_indices:
                if mut_idx in possible_indices:
                    possible_indices.remove(mut_idx)
            continue

    logging.warning(f"Could not find a valid mutation after {attempts} attempts for sequence starting {current_sequence[:10]}...")
    return None


# --- Main Generation Function (Modified) ---
def generate_mutations(
    reference_sequence,
    model,
    tokenizer,
    num_total_variants,
    max_mutations,
    top_k,
    output_dir,
    input_device,
    skip_existing,
    batch_size,
    reference_name="reference",
    reference_stem=None,
):
    """
    Generates a total of `num_total_variants` for the reference sequence,
    distributing them across mutation counts from 1 to `max_mutations`.
    If max_mutations is 1, only single mutants are generated.
    Saves output to ``output_dir`` which will contain ``variants.json`` and
    per-variant FASTA files named ``<reference stem>_esm_<index>.fasta``.
    """
    output_dir_path = Path(output_dir)
    json_path = output_dir_path / "variants.json"
    if skip_existing and json_path.exists():
        logging.info(
            f"Skipping generation as output directory already contains: {json_path}"
        )
        return json_path, 0  # Indicate 0 new variants generated

    fasta_dir_path = output_dir_path

    reference_stem = reference_stem or reference_name or "reference"

    all_variants = []
    variants_generated_count = 0

    logging.info(f"Evaluating up to {batch_size} mutation site(s) per forward pass.")

    # Determine how many variants to generate for each mutation level (1 to max_mutations)
    variants_per_level = defaultdict(int)
    if max_mutations <= 0:
        logging.error("max_mutations must be positive.")
        return None, 0
    if num_total_variants < 0:
        logging.error("num_variants must not be negative.")
        return None, 0
    if batch_size <= 0:
        logging.error("batch_size must be positive.")
        return None, 0

    # Distribute variants somewhat evenly, prioritizing lower mutation counts if not perfectly divisible
    base_count = num_total_variants // max_mutations
    remainder = num_total_variants % max_mutations
    for i in range(1, max_mutations + 1):
        variants_per_level[i] = base_count + (1 if i <= remainder else 0)

    logging.info(f"Targeting variants per mutation level for {reference_name}: {dict(variants_per_level)}")

    total_attempts = 0
    max_total_attempts = num_total_variants * 10 # Global attempt limit

    for num_mut in range(1, max_mutations + 1):
        target_count_for_level = variants_per_level[num_mut]
        if target_count_for_level == 0:
            continue # Skip if no variants are allocated to this level

        generated_for_level = 0
        attempts_for_level = 0
        # Allow more attempts per variant for higher mutation counts
        max_attempts_for_level = target_count_for_level * (5 + num_mut * 2)

        logging.info(f"Generating {target_count_for_level} variants with {num_mut} mutation(s)...")

        # Use a set to avoid duplicate sequences at the same mutation level
        sequences_at_this_level = set()

        while generated_for_level < target_count_for_level and attempts_for_level < max_attempts_for_level and total_attempts < max_total_attempts:
            attempts_for_level += 1
            total_attempts += 1
            current_sequence = reference_sequence
            mutation_history = []
            mutated_positions = set() # Track positions mutated within this chain
            successful_variant = True

            for step in range(num_mut):
                mutation_result = perform_one_mutation(
                    current_sequence,
                    model,
                    tokenizer,
                    top_k,
                    input_device,
                    mutated_positions,
                    batch_size=batch_size,
                )

                if mutation_result:
                    new_sequence, pos, orig_res, new_res = mutation_result
                    # If identity chosen (orig == new) and allowed, treat as zero-mutation variant and stop
                    if orig_res == new_res:
                        successful_variant = True
                        current_sequence = new_sequence  # same as reference
                        # Do NOT add to mutation_history so num_mutations stays 0
                        break
                    # Otherwise, record actual mutation
                    current_sequence = new_sequence
                    mutated_positions.add(pos) # Add 0-based index
                    mutation_history.append({
                        "step": step + 1,
                        "position": pos + 1, # Store 1-based index for output
                        "original_residue": orig_res,
                        "new_residue": new_res
                    })
                else:
                    logging.warning(f"Failed to complete {num_mut}-mutation variant for {reference_name} at step {step+1}.")
                    successful_variant = False
                    break # Stop trying for this specific variant chain

            if successful_variant and current_sequence not in sequences_at_this_level:
                sequences_at_this_level.add(current_sequence)
                variant_name = f"variant_{variants_generated_count+1}_muts{num_mut}_pos" + "_".join(map(str, sorted([h['position'] for h in mutation_history])))
                all_variants.append({
                    "name": variant_name,
                    "sequence": current_sequence,
                    # Record actual number of mutations performed
                    "num_mutations": len(mutation_history),
                    "mutation_history": mutation_history
                })
                generated_for_level += 1
                variants_generated_count += 1
                logging.debug(f"Successfully generated variant {variant_name}")
            elif successful_variant:
                logging.debug(f"Skipping duplicate sequence for variant with {num_mut} mutations.")


        if generated_for_level < target_count_for_level:
             logging.warning(f"Only generated {generated_for_level}/{target_count_for_level} unique variants with {num_mut} mutations for {reference_name} within attempt limits.")

    if variants_generated_count < num_total_variants:
        logging.warning(f"Total unique variants generated for {reference_name}: {variants_generated_count}/{num_total_variants}")

    # If nothing was generated and allowed, emit identity once
    if variants_generated_count == 0:
        logging.warning("No unique variants generated; returning empty variant set.")

    # Save results
    output_data = {
        "reference": {
            "name": reference_name,
            "sequence": reference_sequence
        },
        "variants": all_variants
    }

    try:
        output_dir_path.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(output_data, f, indent=4)
        logging.info(f"Saved {len(all_variants)} variants to {json_path}")

        # --- Write FASTA files ---
        fasta_dir_path.mkdir(parents=True, exist_ok=True)

        # Remove any stale variant FASTA outputs before writing new ones.
        for pattern in (f"{reference_stem}_esm_*.fasta", f"{reference_stem}__esm_*.fasta", "variants_*.fasta"):
            for stale_fasta in fasta_dir_path.glob(pattern):
                try:
                    stale_fasta.unlink()
                except FileNotFoundError:
                    continue

        desired_total = max(num_total_variants, 1)
        generated_variant_count = len(all_variants)

        candidates = [{"type": "variant", "record": var} for var in all_variants]
        if len(candidates) < desired_total:
            candidates.append({"type": "reference", "record": None})
        if not candidates:
            candidates = [{"type": "reference", "record": None}]

        selected_candidates = candidates[:desired_total]
        if len(selected_candidates) < desired_total:
            logging.warning(
                "Requested %s sequences but only %s available (variants + reference).",
                desired_total,
                len(selected_candidates),
            )

        selected_variants = [c["record"] for c in selected_candidates if c["type"] == "variant"]
        dropped_variants = generated_variant_count - len(selected_variants)
        if dropped_variants > 0:
            logging.info(
                "Discarded %s excess variant(s) to match requested total of %s sequences.",
                dropped_variants,
                desired_total,
            )

        all_variants = selected_variants
        output_data["variants"] = all_variants

        for idx, candidate in enumerate(selected_candidates):
            fasta_header = f"{reference_stem}_esm_{idx}"
            fasta_file = fasta_dir_path / f"{fasta_header}.fasta"

            if candidate["type"] == "variant":
                sequence = candidate["record"]["sequence"]
            else:
                sequence = reference_sequence

            with open(fasta_file, "w") as fh:
                fh.write(f">{fasta_header}\n")
                fh.write(sequence)
                fh.write("\n")
            logging.info(f"Wrote FASTA file to {str(fasta_file)}")

            if candidate["type"] == "variant":
                candidate["record"]["fasta_header"] = fasta_header
                candidate["record"]["fasta_path"] = str(fasta_file)

        if not selected_variants and selected_candidates and selected_candidates[0]["type"] == "reference":
            logging.info("No variants generated; emitted reference sequence as _esm_0")

        return json_path, len(all_variants)
    except IOError as e:
        logging.error(f"Failed to write output file {json_path}: {e}")
        return None, 0
    except Exception as e:
        logging.error(f"An unexpected error occurred during file saving: {e}")
        return None, 0

def load_sequence_from_fasta(fasta_file):
    """Loads the first sequence from a FASTA file."""
    if not BIOPYTHON_AVAILABLE:
        logging.error("Cannot load from FASTA file. Biopython library not installed.")
        logging.error("Install using: pip install biopython")
        return None, None

    try:
        with open(fasta_file, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                # Basic validation for protein sequence characters
                seq_str = str(record.seq).upper()
                if not seq_str:
                    logging.warning(f"Skipping empty sequence for record {record.id} in {fasta_file}")
                    continue
                if all(c in 'ACDEFGHIKLMNPQRSTVWYXBZJUO' for c in seq_str):
                     logging.info(f"Loaded sequence ID: {record.id} (Length: {len(seq_str)}) from {fasta_file}")
                     return record.id, seq_str # Return ID and sequence
                else:
                     logging.warning(f"Skipping record {record.id} due to non-standard characters in sequence.")
                     return None, None # Indicate failure due to invalid sequence
            # If loop finishes without returning, no valid sequences found
            logging.error(f"No valid protein sequences found in {fasta_file}")
            return None, None
    except FileNotFoundError:
        logging.error(f"Error: FASTA file not found at {fasta_file}")
        return None, None
    except Exception as e:
        logging.error(f"Error reading FASTA file {fasta_file}: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Generate protein variants using an ESM-2 model. Supports iterative multi-mutation generation.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--ref_sequence", type=str, help="Reference protein sequence as a string.")
    input_group.add_argument("--ref_fasta_file", type=str, help="Path to a FASTA file containing the reference sequence (uses the first sequence found).")

    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output FASTA and JSON files.")
    parser.add_argument("--num_variants", type=int, default=10, help="Total number of variants to generate.")
    parser.add_argument("--max_mutations", type=int, default=1, help="Maximum number of mutations to introduce iteratively. Default is 1 (single mutants only). Set > 1 for iterative multi-mutation variants.")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D", help="Name of the Hugging Face ESM model (e.g., esm2_t33_650M_UR50D, esm2_t48_15B_UR50D).")
    parser.add_argument("--top_k", type=int, default=5, help="Sample from the top-k predictions for each masked position.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of candidate mutation positions evaluated per forward pass.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Path to Hugging Face cache directory.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip generation if output_dir already contains results.")

    args = parser.parse_args()

    # --- Input Validation ---
    reference_sequence = None
    reference_name = "cli_sequence" # Default name if sequence is from CLI

    reference_stem = None

    if args.ref_sequence:
        reference_sequence = args.ref_sequence.upper()
        if not reference_sequence.isalpha() or not all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in reference_sequence):
             logging.error("Invalid characters in --ref_sequence. Only standard amino acids allowed.")
             sys.exit(1)
        reference_stem = reference_name
    elif args.ref_fasta_file:
        fasta_path = Path(args.ref_fasta_file)
        if not fasta_path.is_file():
            logging.error(f"FASTA file not found: {args.ref_fasta_file}")
            sys.exit(1)
        ref_id, ref_seq = load_sequence_from_fasta(fasta_path)
        if ref_seq is None:
            logging.error(f"Could not load a valid sequence from {args.ref_fasta_file}")
            sys.exit(1)
        reference_sequence = ref_seq
        reference_name = ref_id if ref_id else f"seq_from_{fasta_path.stem}"
        reference_stem = fasta_path.stem

    if not reference_sequence: # Should not happen if logic above is correct, but check anyway
        logging.error("Failed to obtain a reference sequence.")
        sys.exit(1)

    if reference_stem is None:
        reference_stem = reference_name

    output_dir_path = Path(args.output_dir)

    if args.num_variants < 0:
        logging.error("--num_variants must be non-negative.")
        sys.exit(1)
    if args.max_mutations <= 0:
         logging.error("--max_mutations must be positive.")
         sys.exit(1)
    if args.top_k <= 0:
        logging.error("--top_k must be positive.")
        sys.exit(1)
    if args.batch_size <= 0:
        logging.error("--batch_size must be positive.")
        sys.exit(1)

    # --- Load Model and Tokenizer (with accelerate support) ---
    logging.info(f"Loading model: {args.model_name}...")
    logging.info(f"Using cache directory: {args.cache_dir if args.cache_dir else 'default'}")
    logging.info("Attempting to load with automatic device map (requires accelerate library)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            device_map="auto", # Automatically distribute model layers
            # Consider torch_dtype=torch.float16 for very large models if needed
            # torch_dtype=torch.float16
        )
        logging.info(f"Model loaded successfully with device map: {model.hf_device_map}")

    except ImportError:
        logging.error("Error: The 'accelerate' library is required for device_map='auto'.")
        logging.error("Please install it using: pip install accelerate")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading model {args.model_name} with device_map='auto': {e}")
        logging.error("Please ensure the model name is correct, you have internet access, accelerate is installed, and necessary dependencies (like sentencepiece) are present.")
        if args.cache_dir:
            logging.error(f"Also check write permissions for the cache directory: {args.cache_dir}")
        if "out of memory" in str(e).lower():
             logging.error("Potential out-of-memory issue during model loading. Check CPU RAM and GPU RAM availability.")
        sys.exit(1)

    # Determine device for placing inputs (usually the first device in the map or CPU)
    try:
      # Heuristic: Use cuda:0 if present in map, otherwise cpu
      if any(d.startswith("cuda") for d in model.hf_device_map.values()):
          input_device = torch.device("cuda:0") # Inputs typically start on cuda:0
      else:
          input_device = torch.device("cpu")
    except Exception:
         input_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Fallback

    logging.info(f"Placing input tensors on device: {input_device}")

    model.eval() # Set model to evaluation mode

    # --- Generate Variants ---
    try:
        generate_mutations(
            reference_sequence=reference_sequence,
            model=model,
            tokenizer=tokenizer,
            num_total_variants=args.num_variants,
            max_mutations=args.max_mutations,
            top_k=args.top_k,
            output_dir=output_dir_path,
            input_device=input_device,
            skip_existing=args.skip_existing,
            batch_size=args.batch_size,
            reference_name=reference_name,
            reference_stem=reference_stem,
        )
    except Exception as e:
        logging.error(f"A critical error occurred during variant generation: {e}")
        # Potentially log traceback here if needed for debugging
        sys.exit(1)

    logging.info("Variant generation process finished.")


if __name__ == "__main__":
    main() 
