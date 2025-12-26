from __future__ import annotations

from typing import Any, Dict, Optional

from profile.utils.denovo_locator import resolve_denovo_artifacts
from profile.utils.samples import lookup_sample


def apply_rules(combo: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    c = dict(combo)
    sample_id = c.get("sample_id") or f"{c.get('scaffold')}_{c.get('ligand')}"
    c["sample_id"] = sample_id
    meta = lookup_sample(sample_id)
    quality = c.get("quality")
    if quality == "low":
        c["step"] = 100
        c["cycle"] = 2
    elif quality == "medium":
        c["step"] = 200
        c["cycle"] = 4
    elif quality == "high":
        c["step"] = 300
        c["cycle"] = 8
    else:
        return None

    try:
        artifacts = resolve_denovo_artifacts(sample_id, scaffold=c.get("scaffold"), ligand=c.get("ligand"))
        if not artifacts.msa_a3m or not artifacts.run_fasta_esm:
            raise FileNotFoundError("Missing MSA or FASTA for protenix")
        c["msa_path"] = str(artifacts.msa_a3m)
        c["fasta_path"] = str(artifacts.run_fasta_esm)
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] protenix input resolution failed for {sample_id}: {exc}")
        return None

    return c


def build_apply_rules():
    return apply_rules
