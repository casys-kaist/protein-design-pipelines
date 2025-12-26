/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    AutoDock Vina docking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
process CONVERT_PDB_TO_PDBQT {
    tag "$meta.id"

    container "vina_cpu_modi:latest"
    memory 1.GB

    input:
        tuple val(meta), path(pdb)
    
    output:
        tuple val(meta), path("${pdb.baseName}_ligand.pdbqt"), path("${pdb.baseName}_receptor.pdbqt"), emit: pdbqt
        path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    args = task.ext.args ?: ''

    ligand_sdf = meta.ligand


    """
    micromamba run -n base python /workspace/scripts/convert_to_pdbqt.py \
        --in_file ${ligand_sdf} \
        --out_pdbqt ${pdb.baseName}_ligand.pdbqt \
        --mode ligand \
        $args

    micromamba run -n base python /workspace/scripts/convert_to_pdbqt.py \
        --in_file ${pdb} \
        --out_pdbqt ${pdb.baseName}_receptor.pdbqt \
        --mode receptor \
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        autodock_vina: \$(git -C /workspace rev-parse HEAD)
END_VERSIONS
    """

    stub:
    """
    touch ${pdb.baseName}_ligand.pdbqt
    touch ${pdb.baseName}_receptor.pdbqt
    """
}

process RUN_AUTODOCK {
    tag "$meta.id"

    container "autodock_vina_cpu:latest"
    memory 1.GB

    input:
        tuple val(meta), path(ligand_pdbqt), path(receptor_pdbqt) 

    output:
        tuple val(meta), path("results/docking/${receptor_pdbqt.baseName}_${ligand_pdbqt.baseName}.pdb"), emit: pdb
        path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    CURRENT_DIR=\$(pwd)

    micromamba run -n base python /workspace/autodock-cpu/inference.py \
        --input_pdbqt_path ${receptor_pdbqt} \
        --ligand_pdbqt_path ${ligand_pdbqt} \
        --output_dir \$CURRENT_DIR \
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        autodock_vina: \$(git -C /workspace rev-parse HEAD)
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

workflow AUTODOCK {
    take:
        ch_pdb

    main:
    ch_versions = Channel.empty()
    CONVERT_PDB_TO_PDBQT(
        ch_pdb,
    )
    ch_versions = ch_versions.mix(CONVERT_PDB_TO_PDBQT.out.versions)

    RUN_AUTODOCK(
        CONVERT_PDB_TO_PDBQT.out.pdbqt
    )
    ch_versions = ch_versions.mix(RUN_AUTODOCK.out.versions)

    emit:
    pdb      = RUN_AUTODOCK.out.pdb
    versions = ch_versions
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
