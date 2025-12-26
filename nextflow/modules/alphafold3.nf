process RUN_ALPHAFOLD3 {
    tag "$meta.id"

    // accelerator 4, type: 'aliyun.com/gpu-mem'
    accelerator 1, type: 'aliyun.com/gpu-count'
    // maxForks 1

    label 'gpu'

    container "alphafold3:latest"
    // Host path mounted into the container; configure via params.alphafold3_model_dir
    def af3_modelpath = params.alphafold3_model_dir

    input:
        tuple val(meta), path(fasta), path(a3m)
        val(num_samples)

    output:
        tuple val(meta), path("results/**/*.cif"), emit: cif
        path "versions.yml", emit: versions

    when:
        task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''

    """
    CURRENT_DIR=\$(pwd)

    mkdir -p alphafold3_inputs results

    python /workspace/scripts/generate_alphafold3_input.py \
        "${a3m}" \
        -f "${fasta}" \
        -o alphafold3_inputs \
        -n "${num_samples}" \
        $args

    (cd /workspace/third_parties/alphafold3 && python run_alphafold.py \
        --norun_data_pipeline \
        --json_path=\$CURRENT_DIR/alphafold3_inputs/alphafold3_input.json \
        --model_dir=$af3_modelpath \
        --output_dir=\$CURRENT_DIR/results \
        $args)

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        alphafold3: null
END_VERSIONS
    """

    stub:
    """
    mkdir -p results/dummy
    touch results/dummy/dummy.cif
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        alphafold3: null
END_VERSIONS
    """
}

process CONVERT_CIF_TO_PDB {
    tag "$meta.id"

    container "autodock_vina_cpu:latest"

    input:
        tuple val(meta), path(cif)

    output:
        tuple val(meta), path("${cif.baseName}.pdb"), emit: pdb

    script:
    """
    # Prefer explicit micromamba root prefix to avoid HOME-based path issues
    if command -v micromamba >/dev/null 2>&1; then
        micromamba -r /root/.local/share/mamba run -n base \
            obabel \
                -icif "${cif}" \
                -O "${cif.baseName}.pdb"
    else
        obabel -icif "${cif}" -O "${cif.baseName}.pdb"
    fi
    """

    stub:
    """
    touch "${cif.baseName}.pdb"
    """
}


workflow ALPHAFOLD3 {

    take:
        ch_fasta_a3m

    main:
        RUN_ALPHAFOLD3(
            ch_fasta_a3m,
            Channel.value(params.alphafold3_num_samples)
        )

        CONVERT_CIF_TO_PDB(
            RUN_ALPHAFOLD3.out.cif.transpose()
        )

    emit:
        versions = RUN_ALPHAFOLD3.out.versions
        pdb = CONVERT_CIF_TO_PDB.out.pdb
}

 /*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
