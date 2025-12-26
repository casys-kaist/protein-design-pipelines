#!/usr/bin/env nextflow
// Adapted from nf-core/proteinfold

nextflow.enable.dsl = 2

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT FUNCTIONS / MODULES / SUBWORKFLOWS / WORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
    include { ESM         } from '../modules/esm'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    NAMED WORKFLOWS FOR PIPELINE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

//
// WORKFLOW: Run main analysis pipeline
//

workflow VARIANT_GEN {
    take: 
        ch_fasta

    main:
    ch_multiqc_files  = Channel.empty()
    ch_versions = Channel.empty()
    if (params.debug_channels) {
        ch_fasta.view { it -> "ESM_IN: ${it[0].id} -> ${it[1].name}" }
    }

    //
    // WORKFLOW: Run ESM variant generation
    //
    ch_variant_json  = Channel.empty()
    ch_variant_fasta = ch_fasta

    if (params.variant_generation_mode == "esm") {
        def targetSamples = params.protenix_num_samples ? params.protenix_num_samples as int : 1
        def fallbackVariants = Math.max(targetSamples - 1, 0)
        def configuredVariants = params.esm_num_variants != null ? params.esm_num_variants as int : null
        def variantsToGenerate = configuredVariants != null ? configuredVariants : fallbackVariants
        variantsToGenerate = Math.max(variantsToGenerate, 0)
        ESM (
            ch_fasta,
            variantsToGenerate,
            params.esm_max_mutations as int,
            params.esm_top_k as int,
        )
        ch_multiqc_files = ESM.out.multiqc
        ch_versions      = ESM.out.versions
        ch_variant_json  = ESM.out.variant_json
        ch_variant_fasta = ESM.out.fasta
        if (params.debug_channels) {
            ch_variant_fasta.view { it -> "ESM_OUT: ${it[0].id} -> ${it[1].name}" }
        }
    }
    emit:
    multiqc        = ch_multiqc_files  // channel: /path/to/multiqc_report.html
    versions       = ch_versions // channel: [version1, version2, ...]
    variant_json   = ch_variant_json
    fasta          = ch_variant_fasta
}
/*
workflow main_run {

    ch_fasta = Channel.fromPath(params.input_fasta).map { path -> [[id: path.baseName], path] }

    VARIANT_GEN(
        ch_fasta
    )
}
*/
