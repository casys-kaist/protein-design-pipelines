/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
/*
process PREPARE_DIFFDOCK_INPUTS {
    tag "$meta.id"
    label 'process_medium'

    container "diffdock:latest"

    input:
        tuple val(meta), path(pdb_dir)

    output:
        tuple val(meta), path("${pdb_dir}/diffdock_inputs.csv"), emit: diffdock_csv
        path "versions.yml", emit: versions

    when:
        task.ext.when == null || task.ext.when

    script:
        args = task.ext.args ?: ''

    ligand_sdf = meta.ligand

    """
        micromamba run -n diffdock python /workspace/scripts/prepare_diffdock_input.py \
            --protein_dir ${pdb_dir} \
            --output_csv ${pdb_dir}/diffdock_inputs.csv \
            --ligand_sdf ${ligand_sdf} \
            $args

        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            diffdock: null
END_VERSIONS
    """

    stub:
    """
        touch ${pdb_dir}/diffdock_inputs.csv

        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            diffdock: null
END_VERSIONS
    """
}
*/

process RUN_DIFFDOCK {
    tag "$meta.id"

    // accelerator 4, type: 'aliyun.com/gpu-mem'
    // accelerator 1, type: 'aliyun.com/gpu-count'
    // maxForks 1

    accelerator 1, type: 'aliyun.com/gpu-count'
    // maxForks 1
    label 'gpu'

    container "diffdock:latest"

    input:
    tuple val(meta), path(pdb)

    output:
    tuple val(meta), path(pdb), path("results/rank1.sdf"), emit: protein_ligand
    path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''

    ligand_sdf = meta.ligand_sdf
    ligand_pdbqt = meta.ligand_pdbqt
    ligand_mol2 = meta.ligand_mol2

    """
    CURRENT_DIR=\$(pwd)

    (cd /workspace/third_parties/DiffDock && micromamba run -n diffdock python3 -m inference \
        --protein_path \$CURRENT_DIR/${pdb} \
        --ligand_description ${ligand_mol2} \
        --out_dir \$CURRENT_DIR \
        --complex_name results \
        $args)

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        diffdock: null
END_VERSIONS
    """

    stub:
    """
    mkdir results

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        diffdock: null
END_VERSIONS
    """
}

workflow DIFFDOCK {

    take:
        ch_pdb

    main:
    ch_versions = Channel.empty()

    RUN_DIFFDOCK(
        ch_pdb
    )
    ch_versions = ch_versions.mix(RUN_DIFFDOCK.out.versions)

    emit:
    protein_ligand        = RUN_DIFFDOCK.out.protein_ligand
    versions       = ch_versions
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/