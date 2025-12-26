from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from profile.config import PATHS
from profile.utils.denovo_locator import resolve_denovo_artifacts
from profile.utils.samples import lookup_sample


def _resolve_ligand_path(ligand: Optional[str], dataset: Optional[str]) -> Optional[str]:
    """Resolve ligand mol2 path for handpicked dataset; fallback to legacy CASF layout."""
    if not ligand:
        return None

    dataset_normalized = (dataset or "").lower()
    if dataset_normalized == "handpicked":
        base = getattr(PATHS.inputs, "handpicked_ligands_mol2", None)
        return str(Path(base) / f"{ligand}.mol2") if base else None

    # Legacy CASF layout (per-complex subdir with *_ligand.mol2).
    base = getattr(PATHS.inputs, "casf_coreset", None)
    if base:
        return str(Path(base) / ligand / f"{ligand}_ligand.mol2")
    return None


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
        c["steps"] = 10
    elif quality == "medium":
        c["steps"] = 20
    elif quality == "high":
        c["steps"] = 30
    else:
        return None

    artifacts = resolve_denovo_artifacts(
        sample_id,
        scaffold=c.get("scaffold") or meta.get("scaffold"),
        ligand=c.get("ligand") or meta.get("ligand"),
        dataset=dataset,
    )

    protein_path = str(artifacts.convert_pdb or artifacts.run_pdb)

    c["protein_path"] = protein_path

    ligand_path = _resolve_ligand_path(c.get("ligand"), dataset)
    if ligand_path is None:
        raise FileNotFoundError(
            f"Could not resolve ligand path for sample_id={sample_id} dataset={dataset} ligand={c.get('ligand')}"
        )
    if not Path(ligand_path).exists():
        raise FileNotFoundError(
            f"Ligand file not found at {ligand_path} (sample_id={sample_id}, dataset={dataset})"
        )
    c["ligand_path"] = ligand_path

    return c


def build_apply_rules():
    return apply_rules
