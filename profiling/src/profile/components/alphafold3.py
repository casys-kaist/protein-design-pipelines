from __future__ import annotations

from typing import Any, Dict, Optional

from profile.utils.denovo_locator import resolve_denovo_artifacts
from profile.utils.samples import lookup_sample


def apply_rules(combo: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    c = dict(combo)
    sample_id = c.get("sample_id") or f"{c.get('scaffold')}_{c.get('ligand')}"
    c["sample_id"] = sample_id
    meta = lookup_sample(sample_id)
    if c.get("quality") == "low":
        c["num_diffusion_steps"] = 100
        c["num_recycles"] = 5
    elif c.get("quality") == "medium":
        c["num_diffusion_steps"] = 200
        c["num_recycles"] = 10
    elif c.get("quality") == "high":
        c["num_diffusion_steps"] = 300
        c["num_recycles"] = 15
    else:
        return None

    try:
        artifacts = resolve_denovo_artifacts(sample_id, scaffold=c.get("scaffold"), ligand=c.get("ligand"))
        if not artifacts.msa_a3m or not artifacts.run_fasta:
            raise FileNotFoundError("Missing MSA or FASTA for alphafold3")
        c["msa_path"] = str(artifacts.msa_a3m)
        c["input_fasta"] = str(artifacts.run_fasta)
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] alphafold3 input resolution failed for {sample_id}: {exc}")
        return None

    return c


def build_apply_rules():
    return apply_rules
