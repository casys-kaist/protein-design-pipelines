/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    AutoDock Vina docking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

process RUN_VINA_CPU {
    tag "$meta.id"

    // accelerator 1, type: 'aliyun.com/gpu-count'
    // maxForks 1
    label 'gpu'

    cpus 4
    container "autodock_vina_cpu:latest"
    memory 12.GB

    input:
        tuple val(meta), path(receptor_pdbqt), path(ligand_pdbqt)
        val(exhaustiveness)

    output:
        tuple val(meta), path("*_docked.pdbqt"), emit: pdb
        path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    center_x = meta.center_x
    center_y = meta.center_y
    center_z = meta.center_z
    """
    CURRENT_DIR=\$(pwd)

    micromamba run -n base python /workspace/autodock-cpu/inference.py \
        --input_pdbqt_path ${receptor_pdbqt} \
        --ligand_pdbqt_path ${ligand_pdbqt} \
        --output_dir \$CURRENT_DIR \
        --center_x ${center_x} \
        --center_y ${center_y} \
        --center_z ${center_z} \
        --exhaustiveness ${exhaustiveness} \
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        autodock_vina: null
END_VERSIONS
    """

    stub:
    """
    touch ${receptor_pdbqt.baseName}_${ligand_pdbqt.baseName}.pdb
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        autodock_vina: null
END_VERSIONS
    """
}

workflow VINA_CPU {
    take:
        ch_pdbqt

    main:
    ch_versions = Channel.empty()

    RUN_VINA_CPU(ch_pdbqt, params.vina_cpu_exhaustiveness)
    ch_versions = ch_versions.mix(RUN_VINA_CPU.out.versions)

    emit:
    pdb      = RUN_VINA_CPU.out.pdb
    versions = ch_versions
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
