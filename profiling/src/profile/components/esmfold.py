from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from profile.config import PATHS
from profile.utils.denovo_locator import resolve_denovo_artifacts
from profile.utils.samples import lookup_sample


def _first_sequence_length(fasta_path: Path) -> Optional[int]:
    try:
        length = 0
        with fasta_path.open() as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if length:
                        break
                    continue
                length += len(line.replace(":", ""))
        return length or None
    except Exception:
        return None


def apply_rules(combo: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    c = dict(combo)
    sample_id = c.get("sample_id") or f"{c.get('scaffold')}_{c.get('ligand')}"
    c["sample_id"] = sample_id
    meta = lookup_sample(sample_id)
    quality = c.get("quality")
    if quality == "low":
        c["recycle"] = 2
    elif quality == "medium":
        c["recycle"] = 4
    elif quality == "high":
        c["recycle"] = 8
    else:
        return None

    try:
        artifacts = resolve_denovo_artifacts(sample_id, scaffold=c.get("scaffold"), ligand=c.get("ligand"))
        if not artifacts.run_fasta_esm:
            raise FileNotFoundError("Missing ESM FASTA")
        c["input_fasta"] = str(artifacts.run_fasta_esm)
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] esmfold input resolution failed for {sample_id}: {exc}")
        return None

    try:
        batch_size = int(c.get("batch_size") or c.get("input_batch_size") or 1)
    except (TypeError, ValueError):
        batch_size = c.get("batch_size") or c.get("input_batch_size") or 1
    c["batch_size"] = batch_size

    seq_len = _first_sequence_length(Path(c["input_fasta"]))
    tokens_per_seq = max(int(seq_len or 0), 1)
    c["max_tokens_per_batch"] = max(tokens_per_seq * int(batch_size), 1)

    try:
        output_root = Path(getattr(PATHS.outputs, "esmfold"))
    except Exception:
        output_root = None
    if output_root:
        batch_fasta = output_root / "inputs" / f"{sample_id}_{quality}_bs{batch_size}.fasta"
        c["batched_fasta"] = str(batch_fasta)
    else:
        c["batched_fasta"] = c.get("input_fasta")

    return c


def build_apply_rules():
    return apply_rules
