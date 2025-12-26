process RUN_RFDIFFUSION {
    tag "$meta.id"
    // Approximately 3600 MiB
    accelerator 4, type: 'aliyun.com/gpu-mem'
    // accelerator 1, type: 'aliyun.com/gpu-count'
    // label 'gpu' 
    memory 4.GB

    container "jinwooh/rfdiffusion:latest"

    input:
        tuple val(meta), path(pdb)
        val model_dir
        val num_designs

    output:
        tuple val(meta), path("${pdb.baseName}_*.pdb"), emit: pdb
        tuple val(meta), path("traj/${pdb.baseName}_*.pdb"), emit: traj
        path "versions.yml", emit: versions

    when:
        task.ext.when == null || task.ext.when

    script:
        def args = task.ext.args ?: ''
        contigs = meta.contigs
        """
        CURRENT_DIR=\$(pwd)

        echo "CURRENT_DIR: \$CURRENT_DIR"

        (cd /workspace/third_parties/RFdiffusion && python3.9 scripts/run_inference.py \
            inference.output_prefix=\$CURRENT_DIR/${pdb.baseName} \
            inference.model_directory_path=${model_dir} \
            inference.input_pdb=\$CURRENT_DIR/${pdb} \
            inference.num_designs=${num_designs} \
            contigmap.contigs='${contigs}' \
            potentials.guide_scale=1 \
            potentials.guiding_potentials='["type:substrate_contacts,weight:3,s:1,r_0:8,rep_r_0:5,rep_s:2,rep_r_min:1"]' \
            potentials.substrate=LIG \
            $args)

        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            rfdiffusion: null
END_VERSIONS
        """

    stub:
        """
        touch ${pdb.baseName}_0.pdb
        mkdir -p traj
        touch traj/${pdb.baseName}_0_Xt-1_traj.pdb
        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            rfdiffusion: null
END_VERSIONS
        """
}

workflow RFDIFFUSION {

    take:
        ch_pdb
        ch_model_dir
        ch_num_designs

    main:
        RUN_RFDIFFUSION(
            ch_pdb,
            ch_model_dir,
            ch_num_designs,
        )
        ch_multiqc_files = Channel.empty()

    emit:
        versions = RUN_RFDIFFUSION.out.versions
        pdb      = RUN_RFDIFFUSION.out.pdb.transpose()
        traj     = RUN_RFDIFFUSION.out.traj.transpose()
        multiqc  = ch_multiqc_files
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
