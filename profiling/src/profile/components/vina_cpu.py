from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from profile.utils.denovo_locator import resolve_denovo_artifacts
from profile.utils.samples import lookup_sample


def apply_rules(combo: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    c = dict(combo)
    sample_id = c.get("sample_id") or f"{c.get('scaffold')}_{c.get('ligand')}"
    c["sample_id"] = sample_id
    meta = lookup_sample(sample_id)
    scaffold = meta.get("scaffold") or c.get("scaffold") or sample_id
    ligand = meta.get("ligand") or c.get("ligand") or scaffold
    c.setdefault("scaffold", scaffold)
    dataset = (meta.get("dataset") or "").lower() or "handpicked"
    ligand_file = Path(f"/mnt/nfs/new/bioinformatics/{dataset}/ligands/pdbqt/{ligand}.pdbqt") if ligand else None
    quality = c.get("quality")
    if quality == "low":
        c["exhaustiveness"] = 4
    elif quality == "medium":
        c["exhaustiveness"] = 8
    elif quality == "high":
        c["exhaustiveness"] = 16
    else:
        return None

    try:
        artifacts = resolve_denovo_artifacts(sample_id, scaffold=c.get("scaffold"), ligand=c.get("ligand"))
        center_path = artifacts.receptor_center
        receptor_path = artifacts.receptor_pdbqt
        if not center_path or not receptor_path:
            raise FileNotFoundError(f"Could not resolve receptor files for {sample_id}")
        c["receptor_path"] = str(receptor_path)
        c["x"], c["y"], c["z"] = load_center_xyz(center_path)
        if ligand_file and ligand_file.exists():
            c["ligand_pdbqt_path"] = str(ligand_file)
        elif ligand_file:
            raise FileNotFoundError(f"Ligand file missing: {ligand_file}")
    except Exception as exc:  # pragma: no cover - defensive logging occurs upstream
        print(f"[WARN] vina_cpu input resolution failed for {c.get('scaffold')} {c.get('ligand')}: {exc}")
        return None

    return c


@lru_cache(maxsize=None)
def load_center_xyz(path: str):
    vals: Dict[str, str] = {}
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in {"center_x", "center_y", "center_z"}:
                vals[key] = value
    if not {"center_x", "center_y", "center_z"} <= set(vals):
        raise ValueError(f"Missing key(s) in {path}: {vals.keys()}")
    return vals["center_x"], vals["center_y"], vals["center_z"]


def build_apply_rules():
    return apply_rules
