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
        c["model"] = "facebook/esm2_t30_150M_UR50D"
    elif quality == "medium":
        c["model"] = "facebook/esm2_t33_650M_UR50D"
    elif quality == "high":
        c["model"] = "facebook/esm2_t36_3B_UR50D"
    else:
        return None

    try:
        artifacts = resolve_denovo_artifacts(sample_id, scaffold=c.get("scaffold"), ligand=c.get("ligand"))
        if not artifacts.run_fasta_esm:
            raise FileNotFoundError("Missing ESM FASTA")
        c["ref_fasta"] = str(artifacts.run_fasta_esm)
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] esm input resolution failed for {sample_id}: {exc}")
        return None

    return c


def build_apply_rules():
    return apply_rules
