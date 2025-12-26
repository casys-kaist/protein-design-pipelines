process CONVERT_PDB_TO_PDBQT {
    tag "$meta.id"

    // Leave this for now
    container "autodock_vina_cpu:latest"
    memory 1.GB

    input:
        tuple val(meta), path(pdb)
    
    output:
        tuple val(meta), path("${pdb.baseName}_receptor.pdbqt"), path("${pdb.baseName}_receptor_center.txt"), emit: pdbqt
        path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    args = task.ext.args ?: ''

    """
    micromamba run -n base python2 /root/.local/share/mamba/bin/prepare_receptor4.py \
        -r ${pdb} \
        -o ${pdb.baseName}_receptor.pdbqt \
        -A checkhydrogens \
        -U nphs_lps \
        $args

    micromamba run -n base python /workspace/autodock-cpu/get_center.py \
        --receptor_pdbqt_path ${pdb.baseName}_receptor.pdbqt > ${pdb.baseName}_receptor_center.txt

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        autodock_vina: null
END_VERSIONS
    """

    stub:
    """
    touch ${pdb.baseName}_receptor.pdbqt
    touch ${pdb.baseName}_receptor_center.txt
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        autodock_vina: null
END_VERSIONS
    """
}

process RUN_VINA_GPU {
    tag "$meta.id"


    // accelerator 1, type: 'aliyun.com/gpu-count'
    // maxForks 1
    label 'gpu'
    memory 1.GB
    
    container "fovus/vina-gpu-2.1:autodock-vina-gpu"
    // ??? GiB is the max memory for a GPU
    accelerator 12, type: 'aliyun.com/gpu-mem'
    

    input:
        tuple val(meta), path(receptor_pdbqt), path(ligand_pdbqt)
        val(threads)

    output:
        tuple val(meta), path("${receptor_pdbqt.baseName}_${ligand_pdbqt.baseName}_docked.pdbqt"), emit: pdb
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

    mkdir -p output_ligands/
    /vina/AutoDock-Vina-GPU-2.1/AutoDock-Vina-GPU-2-1 \
        --opencl_binary_path /vina/AutoDock-Vina-GPU-2.1 \
        --receptor ${receptor_pdbqt} \
        --ligand ${ligand_pdbqt} \
        --center_x ${center_x} \
        --center_y ${center_y} \
        --center_z ${center_z} \
        --size_x 25 \
        --size_y 25 \
        --size_z 25 \
        --thread ${threads} \
        $args

    mv ${ligand_pdbqt.baseName}_out.pdbqt ${receptor_pdbqt.baseName}_${ligand_pdbqt.baseName}_docked.pdbqt

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        autodock_vina: null
END_VERSIONS
    """

    stub:
    """
    mkdir -p output_ligands
    touch ${receptor_pdbqt.baseName}_${ligand_pdbqt.baseName}_docked.pdbqt
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        autodock_vina: null
END_VERSIONS
    """
}

workflow VINA_GPU {
    take:
        ch_pdbqt

    main:
    ch_versions = Channel.empty()

    RUN_VINA_GPU(ch_pdbqt, params.vina_gpu_threads)
    ch_versions = ch_versions.mix(RUN_VINA_GPU.out.versions)

    emit:
    pdb      = RUN_VINA_GPU.out.pdb
    versions = ch_versions
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
