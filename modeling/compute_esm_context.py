#!/usr/bin/env python3
"""
Compute scaffold-level ESM entropy and margin features for each sample_id.

The script:
1. Reads sample definitions from profiling/configs/samples.yaml via resolve_scaffold_path.
2. Extracts the scaffold sequence from the canonical scaffold file.
3. Runs a single forward pass of a Hugging Face ESM model on CPU.
4. Computes per-position entropy and logit margin, then summarizes them.
5. Saves one row per sample_id to CSV with the aggregated statistics.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
import torch
import yaml
from transformers import AutoModelForMaskedLM, AutoTokenizer


DEFAULT_FEATURES_CSV = "modeling/features.csv"
DEFAULT_OUTPUT_CSV = "modeling/esm_context_features.csv"
DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"
ESM_FEATURE_COLUMNS = [
    "esm_mean_entropy",
    "esm_max_entropy",
    "esm_std_entropy",
    "esm_mean_margin",
    "esm_min_margin",
    "esm_std_margin",
]

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    "ASX": "B",
    "GLX": "Z",
    "XLE": "J",
    "UNK": "X",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "profiling" / "configs" / "samples.yaml"


def _dedupe(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in seq:
        if item is None or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _normalize_extensions(extensions: Iterable[str]) -> List[str]:
    normed = []
    for ext in extensions:
        if not ext:
            continue
        norm = ext if ext.startswith(".") else f".{ext}"
        normed.append(norm)
    return _dedupe(normed)


def _dataset_root_from_env() -> Path:
    storage_env = os.environ.get("PROFILE_STORAGE_ROOT")
    storage = Path(storage_env).expanduser() if storage_env else Path("/mnt/nfs/new/bioinformatics/profile")
    return storage.parent if storage.parent != storage else storage


def _load_manifest() -> Dict[str, Dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return {}
    data = yaml.safe_load(MANIFEST_PATH.read_text()) or {}
    samples: Dict[str, Dict[str, str]] = {}
    for group in data.get("samples", {}).values():
        if not group:
            continue
        for entry in group:
            sid = entry.get("sample_id")
            if not sid:
                continue
            samples[str(sid)] = {k: str(v) for k, v in entry.items() if v is not None}
    return samples


def lookup_sample(sample_id: str) -> Dict[str, str]:
    return dict(_load_manifest().get(str(sample_id), {}))


def resolve_scaffold_path(
    scaffold: str | None = None,
    *,
    sample_id: str | None = None,
    dataset: str | None = None,
    datasets_root: Path | None = None,
    extensions: Sequence[str] = (".pdb", ".cif", ".fasta", ".fa"),
) -> Path:
    meta = lookup_sample(sample_id) if sample_id else {}
    scaffold = scaffold or meta.get("scaffold") or meta.get("contig")
    dataset = dataset or meta.get("dataset")

    if not scaffold:
        raise ValueError("scaffold is required; provide scaffold arg or ensure samples.yaml defines it")
    if not dataset:
        raise ValueError("dataset is required; provide dataset arg or ensure samples.yaml defines it")

    datasets_root = Path(datasets_root) if datasets_root else _dataset_root_from_env()
    dataset_variants = _dedupe([dataset, str(dataset).lower(), str(dataset).upper()])
    scaffold_dirs = [datasets_root / d / "scaffolds" for d in dataset_variants]
    existing_dirs = [d for d in scaffold_dirs if d.exists()]
    if not existing_dirs:
        tried = ", ".join(str(d) for d in scaffold_dirs)
        raise FileNotFoundError(f"Scaffold directory not found for dataset={dataset}; tried: {tried}")

    names = _dedupe(
        [
            scaffold,
            scaffold.lower(),
            scaffold.upper(),
            sample_id or None,
            sample_id.lower() if sample_id else None,
            sample_id.upper() if sample_id else None,
        ]
    )
    exts = _normalize_extensions(extensions)

    tried_paths = []
    for root in existing_dirs:
        for name in names:
            for ext in exts:
                candidate = root / f"{name}{ext}"
                tried_paths.append(candidate)
                if candidate.exists():
                    return candidate
        for ext in exts:
            matches = sorted(root.glob(f"*{scaffold}*{ext}"))
            if matches:
                return matches[0]

    tried_str = ", ".join(str(p) for p in tried_paths)
    raise FileNotFoundError(
        f"Scaffold file not found for scaffold={scaffold} dataset={dataset}; tried: {tried_str}"
    )


def set_default_cache_env() -> None:
    """Point torch/transformers caches to shared locations if unset."""
    os.environ.setdefault("TORCH_HOME", "/mnt/nfs/new/bioinformatics/cache/torch")
    os.environ.setdefault("HF_HOME", "/mnt/nfs/new/bioinformatics/cache/hf")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/nfs/new/bioinformatics/cache/hf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ESM entropy/margin summaries per sample_id."
    )
    parser.add_argument(
        "--features-csv",
        default=DEFAULT_FEATURES_CSV,
        help="Existing features CSV used to enumerate sample_id values.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_CSV,
        help="Path to write the ESM context feature CSV.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model name or path (EsmForMaskedLM compatible).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference (default: cpu).",
    )
    return parser.parse_args()


def read_fasta_sequence(path: Path) -> str:
    seq_lines: List[str] = []
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq_lines.append(line)
    return "".join(seq_lines)


def _parse_seqres(lines: Iterable[str]) -> str:
    chains: Dict[str, List[str]] = {}
    for line in lines:
        if not line.startswith("SEQRES"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        chain_id = parts[2]
        residues = parts[4:]
        chains.setdefault(chain_id, [])
        for res in residues:
            chains[chain_id].append(THREE_TO_ONE.get(res.upper(), "X"))
    if not chains:
        return ""
    chain, residues = max(chains.items(), key=lambda kv: len(kv[1]))
    return "".join(residues)


def _parse_atom_records(lines: Iterable[str]) -> str:
    """Fallback extraction using ATOM records (PDB-style columns)."""
    seen = set()
    chains: Dict[str, List[str]] = {}
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        if len(line) < 26:
            continue
        resname = line[17:20].strip()
        chain_id = line[21].strip() or "A"
        resseq = line[22:26].strip()
        icode = line[26].strip()
        key = (chain_id, resseq, icode)
        if key in seen:
            continue
        seen.add(key)
        chains.setdefault(chain_id, [])
        chains[chain_id].append(THREE_TO_ONE.get(resname.upper(), "X"))
    if not chains:
        return ""
    chain, residues = max(chains.items(), key=lambda kv: len(kv[1]))
    return "".join(residues)


def _parse_cif_seq_one_letter(lines: List[str]) -> str:
    """Parse sequence from mmCIF entity_poly fields."""
    tags = [
        "_entity_poly.pdbx_seq_one_letter_code_can",
        "_entity_poly.pdbx_seq_one_letter_code",
    ]
    sequences: List[str] = []
    for tag in tags:
        for idx, line in enumerate(lines):
            if not line.startswith(tag):
                continue
            remainder = line[len(tag) :].strip()
            if remainder.startswith(";"):
                seq_lines: List[str] = []
                for sub in lines[idx + 1 :]:
                    if sub.startswith(";"):
                        break
                    seq_lines.append(sub.strip())
                seq = "".join(seq_lines)
            else:
                seq = remainder.strip().strip("'\"")
            seq = seq.replace(" ", "")
            if seq and seq not in {".", "?"}:
                sequences.append(seq)
        if sequences:
            break
    if not sequences:
        return ""
    return max(sequences, key=len)


def load_scaffold_sequence(scaffold_path: Path) -> str:
    suffix = scaffold_path.suffix.lower()
    if suffix in {".fa", ".fasta"}:
        seq = read_fasta_sequence(scaffold_path)
        if not seq:
            raise ValueError(f"No residues found in FASTA {scaffold_path}")
        return seq

    lines = scaffold_path.read_text().splitlines()
    if suffix == ".cif":
        cif_seq = _parse_cif_seq_one_letter(lines)
        if cif_seq:
            return cif_seq

    seq = _parse_seqres(lines)
    if not seq:
        seq = _parse_atom_records(lines)
    if not seq:
        raise ValueError(f"Could not extract sequence from scaffold file: {scaffold_path}")
    return seq


def compute_entropy_and_margin(sequence: str, model, tokenizer, device: torch.device) -> Dict[str, float]:
    encoded = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0]  # (T, V)

    special_ids = tokenizer.all_special_ids
    if special_ids:
        special_tensor = torch.tensor(special_ids, device=device)
        token_mask = ~torch.isin(input_ids[0], special_tensor)
    else:
        token_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=device)

    logits = logits[token_mask]
    if logits.numel() == 0:
        raise ValueError("No valid tokens after removing special tokens.")

    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    entropy = -(probs * log_probs).sum(dim=-1)

    top2_vals, _ = torch.topk(logits, k=2, dim=-1)
    margins = top2_vals[:, 0] - top2_vals[:, 1]

    return {
        "esm_mean_entropy": entropy.mean().item(),
        "esm_max_entropy": entropy.max().item(),
        "esm_std_entropy": entropy.std(unbiased=False).item(),
        "esm_mean_margin": margins.mean().item(),
        "esm_min_margin": margins.min().item(),
        "esm_std_margin": margins.std(unbiased=False).item(),
    }


def main() -> None:
    args = parse_args()
    set_default_cache_env()

    device = torch.device(args.device)
    df_features = pd.read_csv(args.features_csv)
    sample_ids = sorted(df_features["sample_id"].dropna().unique())
    if not sample_ids:
        raise ValueError("No sample_id values found in the provided features CSV.")

    print(f"Loaded {len(sample_ids)} unique sample_id entries from {args.features_csv}.")
    os.makedirs(Path(args.output).parent, exist_ok=True)

    print(f"Loading ESM model {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    rows = []
    for sample_id in sample_ids:
        try:
            scaffold_path = resolve_scaffold_path(sample_id=str(sample_id))
            sequence = load_scaffold_sequence(scaffold_path)
            stats = compute_entropy_and_margin(sequence, model, tokenizer, device)
            row = {"sample_id": sample_id}
            row.update(stats)
            rows.append(row)
            print(f"[{sample_id}] computed ESM features from {scaffold_path} (len={len(sequence)}).")
        except Exception as exc:  # noqa: BLE001
            print(f"[{sample_id}] failed to compute ESM features: {exc}")

    if not rows:
        raise RuntimeError("No ESM context features were computed; see errors above.")

    df_out = pd.DataFrame(rows)
    df_out = df_out[["sample_id"] + ESM_FEATURE_COLUMNS]
    df_out.to_csv(args.output, index=False)
    print(f"Wrote ESM context features for {len(df_out)} samples to {args.output}.")


if __name__ == "__main__":
    main()
