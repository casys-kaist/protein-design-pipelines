from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[4]
PKG_ROOT = REPO_ROOT / "profiling" / "src"
for path in (PKG_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from profile.utils.denovo_locator import resolve_denovo_artifacts
from profile.utils.samples import lookup_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a batched FASTA for mmseqs2 profiling.")
    parser.add_argument(
        "--source",
        "--input-fasta",
        dest="input_fasta",
        help="Source FASTA path. If omitted, --sample-id is used to resolve denovo artifacts.",
    )
    parser.add_argument(
        "--dest",
        "--output-fasta",
        dest="output_fasta",
        required=True,
        help="Destination FASTA path to write.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of sequences to include in the batched FASTA.",
    )
    parser.add_argument(
        "--sample-id",
        help="Optional sample_id to resolve the input FASTA from denovo artifacts.",
    )
    return parser.parse_args()


def _read_fasta(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header: str | None = None
    seq_lines: List[str] = []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_lines)))
                header = line[1:] or "seq"
                seq_lines = []
            else:
                seq_lines.append(line)
    if header is not None:
        records.append((header, "".join(seq_lines)))
    return records


def _unique_name(base: str, seen: set[str]) -> str:
    candidate = base
    suffix = 1
    while candidate in seen:
        candidate = f"{base}_rep{suffix}"
        suffix += 1
    seen.add(candidate)
    return candidate


def _materialize_batch(records: List[Tuple[str, str]], count: int) -> List[Tuple[str, str]]:
    if not records:
        raise ValueError("Source FASTA contained no sequences.")
    target_count = max(1, int(count))
    seen: set[str] = set()
    batched: List[Tuple[str, str]] = []

    for idx in range(target_count):
        header, seq = records[idx % len(records)]
        base_name = header if idx < len(records) else f"{header}_tile{idx // len(records)}"
        name = _unique_name(base_name, seen)
        batched.append((name, seq))
    return batched


def _write_fasta(records: List[Tuple[str, str]], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as handle:
        for header, seq in records:
            handle.write(f">{header}\n{seq}\n")


def main() -> int:
    args = parse_args()
    output_path = Path(args.output_fasta)
    target_count = max(1, int(args.count))

    if args.input_fasta:
        source_path = Path(args.input_fasta)
    elif args.sample_id:
        meta = lookup_sample(args.sample_id)
        artifacts = resolve_denovo_artifacts(
            args.sample_id,
            scaffold=meta.get("scaffold"),
            ligand=meta.get("ligand"),
            dataset=meta.get("dataset"),
        )
        if not artifacts.run_fasta:
            raise FileNotFoundError(f"No run FASTA found for sample_id={args.sample_id}")
        source_path = artifacts.run_fasta
    else:
        raise ValueError("Provide --source/--input-fasta or --sample-id to locate the FASTA.")

    records = _read_fasta(Path(source_path))
    if len(records) > target_count:
        print(f"[prep_mmseqs_batch] trimming {len(records)} → {target_count} sequences")
        records = records[:target_count]

    batched = _materialize_batch(records, target_count)
    _write_fasta(batched, output_path)

    print(
        f"[prep_mmseqs_batch] wrote {len(batched)} sequences to {output_path} "
        f"(source={source_path}, requested={target_count})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
