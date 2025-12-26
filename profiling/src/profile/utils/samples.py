from __future__ import annotations

import yaml
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from profile import PROFILE_ROOT

MANIFEST_PATH = PROFILE_ROOT.parents[1] / "configs" / "samples.yaml"


@lru_cache(maxsize=None)
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
            if sid:
                samples[str(sid)] = {k: str(v) for k, v in entry.items() if v is not None}
    return samples


def lookup_sample(sample_id: str) -> Dict[str, str]:
    """Return manifest metadata for a sample_id, or {} if unknown."""
    return dict(_load_manifest().get(str(sample_id), {}))


__all__ = ["lookup_sample"]
