# Official Repository for Understanding the Performance Behaviors of End-to-End Protein Design Pipelines on GPUs (CAL 2025)

## Table of Contents

- [0. Path configuration](#0-path-configuration)
- [1. Build Docker Images](#1-build-docker-images)
- [2. Setting up the host machine](#2-setting-up-the-host-machine)
    - [2.1. Installing dependencies](#21-installing-dependencies)
    - [2.2. Setting up Kubernetes and GPUShare](#22-setting-up-kubernetes-and-gpushare)
- [3. Setting ligand and scaffold targets](#3-setting-ligand-and-scaffold-targets)
    - [3.1. Setting scaffolds targets](#31-setting-scaffolds-targets)
    - [3.2. Setting ligands targets](#32-setting-ligands-targets)
    - [3.3. Mol2 sanity checks](#33-mol2-sanity-checks)
- [4. Generating contigs and poses](#4-generating-contigs-and-poses)
    - [4.1. Generate docked ligand–scaffold complexes](#41-generate-docked-ligandscaffold-complexes)
    - [4.2. Generate and validate contigs](#42-generate-and-validate-contigs)
- [5. Populating intermediate files](#5-populating-intermediate-files)
    - [5.1. Populating intermediate files for the first RFDiffusion](#51-populating-intermediate-files-for-the-first-rfdiffusion)
- [6. Component-level profiling](#6-component-level-profiling)
- [7. Pipeline-level profiling](#7-pipeline-level-profiling)
- [License](#license)

## 0. Path configuration

This repo expects most inputs/DBs/model weights to live outside git. Configure paths in these places:

### Component-level profiling runner (`profiling/`)

- **Profiling storage root:** set `PROFILE_STORAGE_ROOT` (default: `/mnt/nfs/new/bioinformatics/profile`). This controls:
  - `profile.RAW_DATA_ROOT`   → `${PROFILE_STORAGE_ROOT}/raw`
  - `profile.OUTPUT_ROOT`     → `${PROFILE_STORAGE_ROOT}/scratch`
  - `profile.INPUT_ROOT`      → `${PROFILE_STORAGE_ROOT}/inputs`
  - `profile.DATASET_ROOT`    → parent dir of `PROFILE_STORAGE_ROOT` (default: `/mnt/nfs/new/bioinformatics`)
- **Dataset/model paths:** edit `profiling/configs/paths.yaml` inputs (e.g., `casf_coreset`, `handpicked_ligands_mol2`, `rfdiffusion_models`, `colabfold_db`, `esmfold_params`) if your data/model directories differ.
- **Pipeline intermediates:** the profiler locates denovo outputs under `${DATASET_ROOT}/handpicked/nextflow` by default; keep your Nextflow `--outdir` aligned (or change `PROFILE_STORAGE_ROOT` / `profiling/configs/paths.yaml` accordingly).

### Pipeline-level runner (`nextflow/`)

- **Input files:** update the absolute paths in:
  - `nextflow/samplesheet/test_docking/handpicked.csv`
  - `nextflow/samplesheet/handpicked/*.csv`
- **Model/DB/cache paths:** edit `nextflow/nextflow.config` (or override via CLI), especially:
  - `params.bioinfo_base_dir` (base directory for several defaults)
  - `params.rfdiffusion_model_directory_path`
  - `params.colabfold_db_path` (MMseqs2/ColabFold DB root)
  - `params.esm_cache_dir` (optional; cache for HuggingFace ESM-2 weights)
  - `params.esmfold_params_path` (only if using ESMFold)
  - `params.alphafold3_model_dir` (only if using AlphaFold3)
  - `env.HF_HOME`, `env.TORCH_HOME`, `env.TRANSFORMERS_CACHE` (optional caches)
- **Runtime mounts:** if using `-profile docker` or `-profile k8s`, ensure the host paths above are mounted in `nextflow/nextflow.config` under `profiles.docker.docker.runOptions` / `profiles.k8s.process.pod`.

## 1. Build Docker Images

We provide a script that always produces:

1. `:latest` tag based on the component's Dockerfile in `docker/<component>/Dockerfile`
2. `:profiling` tag by adding `docker/tools/profiling.Dockerfile` on top of `latest` tag

```bash
python docker/build.py --list
python docker/build.py <image>   # builds <image>:profiling (builds <image>:latest if missing)
python docker/build.py all       # builds all images from --list
```

### Adding new components
To add a new component, you need to:
1. Add a new Dockerfile in the `docker/NEW_COMPONENT/` directory.
2. Build it with `python docker/build.py NEW_COMPONENT` (produces `NEW_COMPONENT:profiling`, and builds `NEW_COMPONENT:latest` if missing).

## 2. Setting up the host machine

### 2.1. Installing dependencies
Install `docker`, `minikube`, `NVIDIA Container Toolkit`, and `conda`.

Then, install the dependencies for the project.
```bash
conda create -n protein-design python=3.10
conda activate protein-design
python -m pip install -r requirements.txt
conda install conda-forge::openbabel
sudo apt install dssp
```

### 2.2. Setting up Kubernetes and GPUShare
Follow the instructions in `docs/kubernetes.md`.

## 3. Setting ligand and scaffold targets
The `.cif` and `.pdb` files for the scaffolds and ligands can be found from [PDB Bank](https://www.rcsb.org/).

For demonstration purpose, we picked 3 popular scaffolds and 3 ligands from CASF-2016 coreset.

### 3.1. Setting scaffolds targets
We surveyed for popular choice of well-studied proteins in the literature.

1. 1shg (62 residues)
2. 1n0r (126 residues)
3. 1tim (247 residues)

### 3.2. Setting ligands targets
We picked 3 ligands from CASF-2016 coreset, which corresponds to 1, 50, 99 percentiles of the heavy-atom count.

```bash
python scripts/pick_ligand_percentiles.py /mnt/nfs/new/bioinformatics/casf/CASF-2016/coreset
```

The ligands are:
1. 3dx1 (9 heavy atoms)
2. 4de1 (23 heavy atoms)
3. 3prs (50 heavy atoms)

Place the 3D structure files under `/mnt/nfs/new/bioinformatics/handpicked` as follows:
```text
/mnt/nfs/new/bioinformatics/handpicked
├── ligands
│   ├── mol2
│   │   ├── 3dx1.mol2
│   │   ├── 3prs.mol2
│   │   └── 4de1.mol2
│   ├── pdbqt
│   │   ├── 3dx1.pdbqt
│   │   ├── 3prs.pdbqt
│   │   └── 4de1.pdbqt
│   └── sdf
│       ├── 3dx1.sdf
│       ├── 3prs.sdf
│       └── 4de1.sdf
└── scaffolds
    ├── 1n0r.pdb
    ├── 1shg.pdb
    └── 1tim.pdb
```

### 3.3. Mol2 sanity checks
Use `scripts/check_mol2.py` to quickly flag broken ligand files (parse errors, zero bonds, collapsed coordinates, <=1 heavy atom). Non-zero exit if a file cannot be parsed.

```bash
python scripts/check_mol2.py -r /mnt/nfs/new/bioinformatics/handpicked/ligands/mol2
```

## 4. Generating contigs and poses
RFDiffusion needs two inputs:
1. a docked ligand–scaffold complex
2. a contig string describing which residues to design.

### 4.1. Generate docked ligand–scaffold complexes
The starting pose affects contig quality. Dock ligands and scaffolds with AutoDock Vina CPU. The sample sheet columns are:
- `sample_id`
- `ligand_sdf`, `ligand_pdbqt`, `ligand_mol2`
- `receptor_pdb` (scaffold PDB path; keep the column name for compatibility)

Run docking with an explicit input sheet and output directory:
```bash
cd nextflow
nextflow run main.nf -profile test_docking,k8s \
  --input samplesheet/test_docking/handpicked.csv \
  --outdir /mnt/nfs/new/bioinformatics/handpicked/nextflow
```

Then combine the scaffold and ligand to get the docked ligand–scaffold complex:
```bash
for scaffold in 1shg 1n0r 1tim; do
  for ligand in 3dx1 4de1 3prs; do
    bash scripts/combine_scaffold_and_ligand.sh \
      --base-dir /mnt/nfs/new/bioinformatics/handpicked \
      --scaffold ${scaffold} \
      --ligand ${ligand} \
      --output ${scaffold}_${ligand}
  done
done
```

### 4.2. Generate and validate contigs
The contig is first generated by using LLMs like ChatGPT, then verified iteratively with the help of DSSP.

Validate the contigs with DSSP:
```bash
# Handpicked scaffolds and ligands
python scripts/validate_contigs.py \
    /mnt/nfs/new/bioinformatics/scaffolds/1shg.cif \
    "[5-10/A8-11/3-6/A17-21/4-7/A29-33/4-6/A41-45/3-5/A50-54/5-10]"

python scripts/validate_contigs.py \
    /mnt/nfs/new/bioinformatics/scaffolds/1n0r.cif \
    "[5-10/A1-33/32-34/A34-66/32-34/A67-99/5-10]"

python scripts/validate_contigs.py \
    /mnt/nfs/new/bioinformatics/scaffolds/1tim.cif \
    "[5-10/A4-15/6-12/A16-28/6-12/A29-43/6-12/A44-58/6-12/A59-73/6-12/A74-88/6-12/A89-103/6-12/A104-118/6-12/A119-133/6-12/A134-148/6-12/A149-163/6-12/A164-178/6-12/A179-193/6-12/A194-208/6-12/A209-223/6-12/A224-238/5-10]"
```

## 5. Populating intermediate files

### 5.1. Populating intermediate files for the first RFDiffusion
In order to profile the components other than the first RFDiffusion, populate the intermediate files:

```bash
cd nextflow
for scaffold in 1shg 1n0r 1tim; do
  for ligand in 3dx1 4de1 3prs; do
    base=${scaffold}_${ligand}
    NXF_DEBUG=1 nextflow run main.nf -profile k8s \
      --pipeline denovo_design_ligand \
      --input "samplesheet/handpicked/${base}.csv" \
      --outdir /mnt/nfs/new/bioinformatics/handpicked/nextflow \
      --rfdiffusion_num_designs 1 \
      --proteinmpnn_num_seq_per_target 1 \
      --esm_num_variants 0 \
      --protenix_num_samples 1
  done
done
```

## 6. Component-level profiling
```bash
python profiling/src/profile/cli/run_sweeps.py profiler=level_a experiments=all # nvidia-smi level
python profiling/src/profile/cli/run_sweeps.py profiler=level_b experiments=all # nsight systems level
python profiling/src/profile/cli/run_sweeps.py profiler=level_c experiments=all # nsight compute level
```

At this point, you should be able to plot Figure 2 in the paper:
```bash
python scripts/plot_microbench.py
```

## 7. Pipeline-level profiling

### 7.1. Normalized runtime of pipeline components across varied sampling (Figure 3)
The `nextflow/multiple_runs_launch.py` script constrains the number of GPU counts by launching blocker pods.

```bash
python nextflow/multiple_runs_launch.py \
  --fanout-levels 1 2 3 \
  --collocate-options exclusive \
  --gpu-counts 1 \
  --repeats 3
```

```bash
python scripts/plot_fanout.py
```

### 7.2. GPU temporal utilization over execution time with and without collocation (Figure 4)
```bash
python nextflow/multiple_runs_launch.py \
  --fanout-levels 2 3 \
  --collocate-options collocate \
  --gpu-counts 1 \
  --repeats 3

python nextflow/multiple_runs_launch.py \
  --fanout-levels 5 4 \
  --collocate-options collocate \
  --gpu-counts 1 2 3 4 \
  --repeats 3
```

At this point, you should be able to plot Figure 4(c,d) in the paper:
```bash
python scripts/plot_multi_gpu_panel.py
```

## License
See `LICENSE`. Third-party components under `third_parties/` are governed by their own licenses and terms.

# Citation

If you find this work useful, please cite:

```bibtex
@ARTICLE{hwang2025protein,
author={Hwang, Jinwoo and Hwang, Yeongmin and Meaza, Tadiwos and Bae, Hyeonbin and Park, Jongse},
journal={IEEE Computer Architecture Letters},
title={{Understanding the Performance Behaviors of End-to-End Protein Design Pipelines on GPUs}},
doi={10.1109/LCA.2025.3646250},
year={2025},
```
