#!/usr/bin/env nextflow
// Adapted from nf-core/proteinfold

nextflow.enable.dsl = 2

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT FUNCTIONS / MODULES / SUBWORKFLOWS / WORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

if (params.structure_prediction_mode == "esmfold") {
    include { PREPARE_ESMFOLD_DBS } from './local/prepare_esmfold_dbs'
    include { ESMFOLD             } from '../modules/esmfold'
}
else if (params.structure_prediction_mode == "protenix") {
    include { PROTENIX } from '../modules/protenix'
}
else if (params.structure_prediction_mode == "alphafold3") {
    include { ALPHAFOLD3             } from '../modules/alphafold3'
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    NAMED WORKFLOWS FOR PIPELINE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

//
// WORKFLOW: Run main analysis pipeline
//
workflow STRUCTURE_PREDICTION {
    take:
        ch_fasta_a3m

    main:
    ch_pdbs = Channel.empty()
    ch_multiqc_files  = Channel.empty()
    ch_versions = Channel.empty()

    //
    // WORKFLOW: Run structure prediction
    //
    if (params.structure_prediction_mode == "esmfold") {
        //
        // SUBWORKFLOW: Prepare esmfold DBs
        //
        PREPARE_ESMFOLD_DBS (
            params.esmfold_db,
            params.esmfold_params_path,
            params.esmfold_3B_v1,
            params.esm2_t36_3B_UR50D,
            params.esm2_t36_3B_UR50D_contact_regression
        )
        ch_versions = ch_versions.mix(PREPARE_ESMFOLD_DBS.out.versions)

        //
        // WORKFLOW: Run nf-core/esmfold workflow
        //
        ESMFOLD (
            ch_versions,
            ch_fasta_a3m,
            PREPARE_ESMFOLD_DBS.out.params,
            params.esmfold_num_cycles
        )
        ch_pdbs = ESMFOLD.out.pdb
        ch_versions = ch_versions.mix(ESMFOLD.out.versions)
        ch_multiqc_files = ESMFOLD.out.multiqc
    }
    //
    //
    // WORKFLOW: Run protenix
    //
    else if(params.structure_prediction_mode == "protenix") {
        PROTENIX(
            ch_fasta_a3m
        )
        ch_pdbs = PROTENIX.out.pdb
        ch_versions = ch_versions.mix(PROTENIX.out.versions)
    }
    //
    //
    // WORKFLOW: Run alphafold3
    //
    else if(params.structure_prediction_mode == "alphafold3") {
        ALPHAFOLD3(
            ch_fasta_a3m
        )
        ch_pdbs = ALPHAFOLD3.out.pdb
        ch_versions = ch_versions.mix(ALPHAFOLD3.out.versions)
    }
    if (params.debug_channels) {
        ch_pdbs.view { it -> "PRED_OUT: ${it[0].id} -> ${it[1].name}" }
    }
    emit:
    pdb            = ch_pdbs
    multiqc        = ch_multiqc_files  // channel: /path/to/multiqc_report.html
    versions       = ch_versions // channel: [version1, version2, ...]
}
