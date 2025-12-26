from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

from profile import DATASET_ROOT
from profile.utils.samples import lookup_sample


def _dedupe(seq: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for item in seq:
        if item in seen or item is None:
            continue
        seen.add(item)
        out.append(item)
    return out


def _normalize_extensions(extensions: Iterable[str]) -> list[str]:
    normed = []
    for ext in extensions:
        if not ext:
            continue
        norm = ext if ext.startswith(".") else f".{ext}"
        normed.append(norm)
    return _dedupe(normed)


def resolve_scaffold_path(
    scaffold: Optional[str] = None,
    *,
    sample_id: Optional[str] = None,
    dataset: Optional[str] = None,
    datasets_root: Optional[Path] = None,
    extensions: Sequence[str] = (".pdb", ".cif", ".fasta", ".fa"),
) -> Path:
    """
    Resolve a scaffold structure path for a given dataset.

    By default, searches under:
      <DATASET_ROOT>/<dataset>/scaffolds/<scaffold>.<ext>
    where DATASET_ROOT is derived from PROFILE_STORAGE_ROOT (see layout.py).
    """

    meta = lookup_sample(sample_id) if sample_id else {}
    scaffold = scaffold or meta.get("scaffold") or meta.get("contig")
    dataset = dataset or meta.get("dataset")

    if not scaffold:
        raise ValueError("scaffold is required; provide scaffold arg or ensure samples.yaml defines it")
    if not dataset:
        raise ValueError("dataset is required; provide dataset arg or ensure samples.yaml defines it")

    datasets_root = Path(datasets_root) if datasets_root else Path(DATASET_ROOT)

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
        # Fallback: partial match (useful when only long IDs exist).
        for ext in exts:
            matches = sorted(root.glob(f"*{scaffold}*{ext}"))
            if matches:
                return matches[0]

    tried_str = ", ".join(str(p) for p in tried_paths)
    raise FileNotFoundError(
        f"Scaffold file not found for scaffold={scaffold} dataset={dataset}; tried: {tried_str}"
    )


__all__ = ["resolve_scaffold_path"]
