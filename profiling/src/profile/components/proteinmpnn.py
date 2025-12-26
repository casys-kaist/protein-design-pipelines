from __future__ import annotations

from typing import Any, Dict, Optional

from profile.utils.denovo_locator import resolve_denovo_artifacts


def apply_rules(combo: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    c = dict(combo)
    sample_id = c.get("sample_id") or f"{c.get('scaffold')}_{c.get('ligand')}"
    c["sample_id"] = sample_id
    try:
        artifacts = resolve_denovo_artifacts(sample_id, scaffold=c.get("scaffold"), ligand=c.get("ligand"))
        if not artifacts.run_pdb:
            raise FileNotFoundError("Missing run PDB")
        # Fan out the same PDB path for each requested input (later replace with distinct inputs in real runs).
        batch_size = c.get("batch_size") or c.get("input_batch_size") or 1
        try:
            batch_size = int(batch_size)
        except Exception:
            batch_size = 1
        batch_size = max(1, batch_size)
        paths = [str(artifacts.run_pdb)] * batch_size
        c["protein_path"] = ",".join(paths)
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] proteinmpnn input resolution failed for {sample_id}: {exc}")
        return None
    return c


def build_apply_rules():
    return apply_rules
