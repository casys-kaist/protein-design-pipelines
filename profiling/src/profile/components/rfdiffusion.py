from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path

from profile import DATASET_ROOT
from profile.config import PATHS
from profile.utils.samples import lookup_sample

# Auto-generated from nextflow/samplesheet/**/*.csv (sequence, contigs).
# Keyed by sample_id; values are raw contig strings used by RFdiffusion.
_CONTIG_MAP = {
    "1n0r_3dx1": "[5-10/A1-33/32-34/A34-66/32-34/A67-99/5-10]",
    "1n0r_3prs": "[5-10/A1-33/32-34/A34-66/32-34/A67-99/5-10]",
    "1n0r_4de1": "[5-10/A1-33/32-34/A34-66/32-34/A67-99/5-10]",
    "1shg_3dx1": "[5-10/A8-11/3-6/A17-21/4-7/A29-33/4-6/A41-45/3-5/A50-54/5-10]",
    "1shg_3prs": "[5-10/A8-11/3-6/A17-21/4-7/A29-33/4-6/A41-45/3-5/A50-54/5-10]",
    "1shg_4de1": "[5-10/A8-11/3-6/A17-21/4-7/A29-33/4-6/A41-45/3-5/A50-54/5-10]",
    "1tim_3dx1": "[5-10/A4-15/6-12/A16-28/6-12/A29-43/6-12/A44-58/6-12/A59-73/6-12/A74-88/6-12/A89-103/6-12/A104-118/6-12/A119-133/6-12/A134-148/6-12/A149-163/6-12/A164-178/6-12/A179-193/6-12/A194-208/6-12/A209-223/6-12/A224-238/5-10]",
    "1tim_3prs": "[5-10/A4-15/6-12/A16-28/6-12/A29-43/6-12/A44-58/6-12/A59-73/6-12/A74-88/6-12/A89-103/6-12/A104-118/6-12/A119-133/6-12/A134-148/6-12/A149-163/6-12/A164-178/6-12/A179-193/6-12/A194-208/6-12/A209-223/6-12/A224-238/5-10]",
    "1tim_4de1": "[5-10/A4-15/6-12/A16-28/6-12/A29-43/6-12/A44-58/6-12/A59-73/6-12/A74-88/6-12/A89-103/6-12/A104-118/6-12/A119-133/6-12/A134-148/6-12/A149-163/6-12/A164-178/6-12/A179-193/6-12/A194-208/6-12/A209-223/6-12/A224-238/5-10]",
}


def apply_rules(combo: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    c = dict(combo)
    sample_id = c.get("sample_id") or f"{c.get('scaffold')}_{c.get('ligand')}"
    c["sample_id"] = sample_id
    meta = lookup_sample(sample_id)
    c.setdefault("scaffold", meta.get("scaffold"))
    c.setdefault("ligand", meta.get("ligand"))
    dataset = c.get("dataset") or meta.get("dataset")
    if dataset:
        c["dataset"] = dataset

    quality = c.get("quality")
    if quality == "low":
        c["step"] = 20
    elif quality == "medium":
        c["step"] = 50
    elif quality == "high":
        c["step"] = 100
    else:
        return None

    def _as_positive_int(value: Any, default: Optional[int] = None) -> Optional[int]:
        if value is None:
            return default
        try:
            parsed = int(value)
            return parsed if parsed > 0 else default
        except (TypeError, ValueError):
            return default

    # RFdiffusion does not batch inputs; forward output_samples (or fallback) into num_designs.
    designs = _as_positive_int(c.get("output_samples"))
    if designs is None:
        designs = _as_positive_int(c.get("num_designs"), default=1)
    c["num_designs"] = designs

    default_pdb = Path(DATASET_ROOT) / "docked" / "processed" / f"{sample_id}.pdb"
    candidates = [default_pdb]
    if dataset:
        dataset_pdb = Path(DATASET_ROOT) / str(dataset).lower() / "docked" / f"{sample_id}.pdb"
        candidates.insert(0, dataset_pdb)
    selected = next((path for path in candidates if path.exists()), candidates[0])
    c["input_pdb"] = str(selected)

    scaffold = c.get("scaffold") or c.get("contig")
    # Prefer manifest-provided contig; fallback to internal map keyed by sample_id (case-insensitive).
    contig = meta.get("contig") or _CONTIG_MAP.get(sample_id) or _CONTIG_MAP.get(sample_id.lower())
    if not contig:
        return None
    c["actual_contig"] = contig
    return c


def build_apply_rules():
    return apply_rules
