process RUN_PROTENIX {
    tag "$meta.id"

    accelerator 4, type: 'aliyun.com/gpu-mem'
    // accelerator 1, type: 'aliyun.com/gpu-count'
    // maxForks 1

    label 'gpu'
    memory 44.GB

    container "protenix:latest"

    input:
        tuple val(meta), path(fasta), path(a3m)
        val(num_samples)
        val(n_step)
        val(n_cycle)

    output:
        tuple val(meta), path("results/**/*.cif"), emit: cif
        path "versions.yml", emit: versions

    when:
        task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    CURRENT_DIR=\$(pwd)

    mkdir -p protenix_inputs

    python /workspace/scripts/generate_protenix_input.py \
        "${a3m}" \
        -f "${fasta}" \
        -o protenix_inputs \
        $args

    (cd /workspace/third_parties/Protenix && bash inference_demo.sh \
        --input_json_path \$CURRENT_DIR/protenix_inputs/protenix_input.json \
        --dump_dir \$CURRENT_DIR/results \
        --N_sample ${num_samples} \
        --N_step ${n_step} \
        --N_cycle ${n_cycle} \
        $args)

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        protenix: null
END_VERSIONS
    """

    stub:
    """
    mkdir -p results/dummy
    touch results/dummy/dummy.cif
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        protenix: null
END_VERSIONS
    """
}

process CONVERT_CIF_TO_PDB {
    tag "$meta.id"

    container "autodock_vina_cpu:latest"
    memory 1.GB

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


workflow PROTENIX {

    take:
        ch_fasta_a3m

    main:
        RUN_PROTENIX(
            ch_fasta_a3m,
            params.protenix_num_samples,
            params.protenix_n_step,
            params.protenix_n_cycle
        )

        CONVERT_CIF_TO_PDB(
            RUN_PROTENIX.out.cif.transpose()
        )

    emit:
        versions = RUN_PROTENIX.out.versions
        pdb = CONVERT_CIF_TO_PDB.out.pdb
}

 /*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
