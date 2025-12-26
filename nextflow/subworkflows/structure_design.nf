#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

if (params.structure_design_mode == "rfdiffusion") {
    include { RFDIFFUSION } from '../modules/rfdiffusion'
}

workflow STRUCTURE_DESIGN {
    take:
        ch_pdb

    main:
        ch_multiqc_files = Channel.empty()
        ch_versions = Channel.empty()
        ch_designed = Channel.empty()

        if (params.structure_design_mode == "rfdiffusion") {
            RFDIFFUSION(
                ch_pdb,
                params.rfdiffusion_model_directory_path,
                params.rfdiffusion_num_designs,
            )
            ch_versions = ch_versions.mix(RFDIFFUSION.out.versions)
            ch_designed = RFDIFFUSION.out.pdb
        } else {
            throw new Exception("Structure design mode ${params.structure_design_mode} not supported")
        }

    emit:
        multiqc  = ch_multiqc_files
        versions = ch_versions
        pdb      = ch_designed
}
