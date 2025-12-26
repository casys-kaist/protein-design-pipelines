from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from profile import DATASET_ROOT
from profile.utils.samples import lookup_sample


@dataclass
class DenovoArtifacts:
    base_dir: Path
    convert_dir: Path
    run_dir: Path
    msa_dir: Path
    dataset: Optional[str] = None
    msa_a3m: Optional[Path] = None
    convert_pdb: Optional[Path] = None
    receptor_pdbqt: Optional[Path] = None
    receptor_center: Optional[Path] = None
    run_fasta: Optional[Path] = None
    run_fasta_esm: Optional[Path] = None
    run_pdb: Optional[Path] = None
    docked_pdbqt: Optional[Path] = None


def _find_first(patterns: Iterable[str]) -> Optional[Path]:
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return Path(matches[0])
    return None


def _valid_pdb(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return None
        with path.open("r") as handle:
            for idx, line in enumerate(handle):
                if "ATOM" in line or line.startswith("ATOM"):
                    return path
                if idx > 200:  # quick scan
                    break
    except Exception:
        return None
    return None


def _candidate_bases(sample_id: str, scaffold: Optional[str], ligand: Optional[str], dataset: str) -> List[Path]:
    bases: List[Path] = []
    sample_lower = sample_id.lower()
    sample_upper = sample_id.upper()
    dataset_root = Path(DATASET_ROOT) / str(dataset).lower() / "nextflow"
    bases.append(dataset_root / "denovo" / sample_id)
    bases.append(dataset_root / "denovo" / sample_lower)
    bases.append(dataset_root / "denovo" / sample_upper)
    bases.append(dataset_root / sample_id)
    bases.append(dataset_root / sample_lower)
    bases.append(dataset_root / sample_upper)
    bases.append(dataset_root)  # fallback to dataset root if denovo/<sample> is missing
    if scaffold and ligand:
        bases.append(dataset_root / f"denovo_{scaffold}_{ligand}")
        bases.append(dataset_root / f"{scaffold}_{ligand}")
    # Deduplicate while preserving order.
    unique: List[Path] = []
    seen = set()
    for b in bases:
        key = b.resolve() if b.is_absolute() else b
        if key in seen:
            continue
        seen.add(key)
        unique.append(b)
    return unique


def resolve_denovo_artifacts(
    sample_id: str,
    *,
    scaffold: Optional[str] = None,
    ligand: Optional[str] = None,
    dataset: Optional[str] = None,
) -> DenovoArtifacts:
    """Resolve denovo outputs for a sample_id."""
    meta = lookup_sample(sample_id)
    scaffold = scaffold or meta.get("scaffold")
    ligand = ligand or meta.get("ligand")
    dataset = dataset or meta.get("dataset")
    if not dataset:
        raise ValueError(
            f"Dataset is required to resolve denovo artifacts for sample_id={sample_id}; "
            "ensure samples.yaml includes dataset or pass it explicitly."
        )

    bases = _candidate_bases(sample_id, scaffold, ligand, dataset)
    base_dir = next((b for b in bases if b.exists()), None)
    if base_dir is None:
        tried = ", ".join(str(b) for b in bases)
        raise FileNotFoundError(f"No denovo directory found for {sample_id}; tried: {tried}")

    # Handpicked structure keeps convert/run at the root.
    convert_dir = base_dir / "convert"
    run_dir = base_dir / "run"
    msa_dir = base_dir / "msa"

    prefixes: List[str] = []
    prefixes.append(sample_id)
    prefixes.append(sample_id.lower())
    prefixes.append(sample_id.upper())
    if scaffold and ligand:
        prefixes.append(f"{scaffold}_{ligand}")
        prefixes.append(f"{scaffold.lower()}_{ligand.lower()}")
    prefixes = list(dict.fromkeys(prefixes))  # dedupe

    handpicked = dataset == "handpicked"

    def patterns_for_convert(name_suffix: str) -> List[str]:
        pats: List[str] = []
        for prefix in prefixes:
            pats.append(str(convert_dir / f"{prefix}_0_mpnn_0_esm_0_seed_101_sample_0{name_suffix}"))
            pats.append(str(convert_dir / f"{prefix}_0_mpnn_0_esm_0{name_suffix}"))
            pats.append(str(convert_dir / f"{prefix}_*{name_suffix}"))
        if handpicked and scaffold:
            pats.append(str(convert_dir / f"{scaffold}_receptor{name_suffix}"))
            pats.append(str(convert_dir / f"{scaffold.upper()}_receptor{name_suffix}"))
            pats.append(str(convert_dir / f"{scaffold.lower()}_receptor{name_suffix}"))
        return pats

    def patterns_for_run(name_suffix: str) -> List[str]:
        pats: List[str] = []
        for prefix in prefixes:
            pats.append(str(run_dir / f"{prefix}_0_mpnn_0_esm_0_seed_101_sample_0{name_suffix}"))
            pats.append(str(run_dir / f"{prefix}_*{name_suffix}"))
        if handpicked and scaffold and ligand:
            pats.append(str(run_dir / f"{scaffold}_receptor_{ligand}{name_suffix}"))
            pats.append(str(run_dir / f"{str(scaffold).upper()}_receptor_{str(ligand).upper()}{name_suffix}"))
            pats.append(str(run_dir / f"{str(scaffold).lower()}_receptor_{str(ligand).lower()}{name_suffix}"))
        return pats

    artifacts = DenovoArtifacts(
        base_dir=base_dir,
        convert_dir=convert_dir,
        run_dir=run_dir,
        msa_dir=msa_dir,
        dataset=dataset,
        msa_a3m=_find_first([str(msa_dir / f"{p}_0_mpnn_0_esm_0.a3m") for p in prefixes]),
        convert_pdb=_find_first(patterns_for_convert(".pdb")),
        receptor_pdbqt=_find_first(patterns_for_convert("_receptor.pdbqt")),
        receptor_center=_find_first(patterns_for_convert("_receptor_center.txt")),
        run_fasta=_find_first(
            [str(run_dir / f"{p}_0_mpnn_0.fasta") for p in prefixes]
            + [str(run_dir / f"{p}_0.fasta") for p in prefixes]
        ),
        run_fasta_esm=_find_first([str(run_dir / f"{p}_0_mpnn_0_esm_0.fasta") for p in prefixes]),
        run_pdb=_find_first([str(run_dir / f"{p}_0.pdb") for p in prefixes]),
        docked_pdbqt=_find_first(patterns_for_run("_docked.pdbqt")),
    )
    artifacts.convert_pdb = _valid_pdb(artifacts.convert_pdb)
    artifacts.run_pdb = _valid_pdb(artifacts.run_pdb)
    if artifacts.convert_pdb is None and artifacts.run_pdb is None:
        raise FileNotFoundError(
            f"Missing usable denovo protein structure for sample_id={sample_id} (dataset={dataset}); "
            f"expected readable convert/run PDB under {base_dir}."
        )
    return artifacts


__all__ = ["DenovoArtifacts", "resolve_denovo_artifacts"]
