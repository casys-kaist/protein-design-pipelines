# Nextflow pipelines

Entry point: `nextflow/main.nf`. Adapted from nf-core/proteinfold: https://github.com/nf-core/proteinfold

- End-to-end setup and required host paths: see repo root `README.md`.
- Sample sheets live under `nextflow/samplesheet/`.

## Test profiles
Use `-profile test_<name>,k8s` (recommended) or `-profile test_<name>,docker` (legacy). Profile order matters: test profile first, runtime profile last.

- `test_docking`: docking only; add `--docking_mode vina_gpu` to force GPU.
- `test_msa`: MSA only; `--msa_mode mmseqs2` and `--use_gpu true` for GPU nodes.
- `test_structure_prediction`: MSA + structure prediction; `--structure_prediction_mode protenix|esmfold|colabfold|alphafold3` (models/licenses required where applicable).
- `test_structure_design`: RFdiffusion design; requires a design samplesheet.

## Quick smoke run
```bash
cd nextflow
nextflow run main.nf -profile test_docking,k8s \
  --input samplesheet/test_docking/handpicked.csv \
  --outdir <OUTDIR>
```

## Common overrides
```bash
# Custom input/output
nextflow run main.nf -profile test_msa,k8s \
  --input samplesheet/samplesheet_msa.csv \
  --outdir <OUTDIR>

# CPU vs GPU
nextflow run main.nf -profile test_msa,k8s --use_gpu false
nextflow run main.nf -profile test_msa,k8s --use_gpu true

# Partial pipeline (skip steps)
nextflow run main.nf -profile k8s \
  --input samplesheet.csv \
  --msa_mode null \
  --structure_prediction_mode null \
  --docking_mode null
```

## Input formats
Docking:
```csv
sample_id,ligand_sdf,ligand_pdbqt,scaffold_pdb,ligand_mol2
```
Notes: `scaffold_pdb` can be provided as `receptor_pdb`; `ligand_pdbqt` is required for Vina; `ligand_mol2` is required for DiffDock.

MSA / structure prediction:
```csv
sample_id,fasta_file
```

Structure design:
```csv
sample_id,pdb_file,contigs
```

## Troubleshooting (quick checks)
- Ensure profile order is `test_<name>,k8s` (or `...,docker`).
- Verify K8s connectivity with `kubectl cluster-info`; check logs for `executor > k8s`.
- For Docker runs, confirm `docker ps` works and mounts are correct.
- For GPU runs, use `--use_gpu true` and confirm GPU availability on nodes.

## Preloading MMseqs2 indexes
MMseqs2 is a disk-intensive tool. So we need to preload the indexes before running the pipeline.
```
sudo su
ulimit -l unlimited
cd /mnt/ssd/colabfold_db

# First, assure that we have enough memory
du -ch uniref*.{index,lookup} uniref*_seq  uniref*db uniref*dbtype # 154G
du -ch uniref*.{index,lookup,idx} uniref*_seq  uniref*db uniref*dbtype # 382G

# vmtouch -f -t -l uniref*.{index,lookup} uniref*_seq  uniref*db uniref*dbtype
# If you have more than 382 GB
# vmtouch -f -t -l uniref*.{index,lookup,idx} uniref*_seq  uniref*db uniref*dbtype
```