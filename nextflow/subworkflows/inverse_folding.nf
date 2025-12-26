#!/usr/bin/env nextflow
// Adapted from nf-core/proteinfold

nextflow.enable.dsl = 2

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT FUNCTIONS / MODULES / SUBWORKFLOWS / WORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

if (params.inverse_folding_mode == "proteinmpnn") {
    include { PROTEINMPNN } from '../modules/proteinmpnn'
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    NAMED WORKFLOWS FOR PIPELINE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

//
// WORKFLOW: Run main analysis pipeline
//
workflow INVERSE_FOLDING {
    take: 
        ch_pdb

    main:
    ch_multiqc_files  = Channel.empty()
    ch_versions = Channel.empty()

    //
    // WORKFLOW: Run inverse folding
    //
    ch_fasta = Channel.empty()
    if (params.inverse_folding_mode == "proteinmpnn") {
        PROTEINMPNN (
            ch_pdb,
            params.proteinmpnn_num_seq_per_target,
            params.proteinmpnn_sampling_temp,
        )
        ch_multiqc_files = PROTEINMPNN.out.multiqc
        ch_versions = PROTEINMPNN.out.versions
        ch_fasta = PROTEINMPNN.out.fasta
        if (params.debug_channels) {
            ch_fasta.view { it -> "IF_FASTA: ${it[0].id} -> ${it[1].name}" }
        }
    }
    emit:
    multiqc        = ch_multiqc_files  // channel: /path/to/multiqc_report.html
    versions       = ch_versions // channel: [version1, version2, ...]
    fasta          = ch_fasta
}
