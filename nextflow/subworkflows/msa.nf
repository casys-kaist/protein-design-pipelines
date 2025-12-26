#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT FUNCTIONS / MODULES / SUBWORKFLOWS / WORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { MMSEQS2 } from '../modules/mmseqs2'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    MSA PARAMETER VALUES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

// MSA parameters are now defined in main config files

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    NAMED WORKFLOWS FOR PIPELINE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

//
// WORKFLOW: Run Multiple Sequence Alignment
//
workflow MSA {
    take: 
        ch_fasta

    main:
    ch_msa_files = Channel.empty()
    ch_multiqc_files  = Channel.empty()
    ch_versions = Channel.empty()
    if (params.debug_channels) {
        ch_fasta.view { it -> "MSA_IN: ${it[0].id} -> ${it[1].name}" }
    }

    //
    // WORKFLOW: Run MMseqs2 for MSA generation
    //
    if(params.msa_mode == "mmseqs2") {
        //
        // MODULE: Run MMseqs2
        //
        MMSEQS2 (
            ch_versions,
            ch_fasta,
            params.colabfold_db_path,
            params.mmseqs2_db1,
            params.mmseqs2_db3,
            params.mmseqs2_batch_size,
            params.mmseqs2_num_iterations
        )
        ch_versions = ch_versions.mix(MMSEQS2.out.versions)
        ch_fasta_a3m = MMSEQS2.out.fasta_a3m
        if (params.debug_channels) {
            ch_fasta_a3m.view { it -> "MSA_OUT: ${it[0].id} -> ${it[2].name}" }
        }
    } else {
        error "Unsupported msa_mode='${params.msa_mode}'. Use 'mmseqs2' or set it to 'null' to skip MSA."
    }

    emit:
    fasta_a3m      = ch_fasta_a3m
    multiqc        = ch_multiqc_files  // channel: /path/to/multiqc_report.html
    versions       = ch_versions // channel: [version1, version2, ...]
} 
