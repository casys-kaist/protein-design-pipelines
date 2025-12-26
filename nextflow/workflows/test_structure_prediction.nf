#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { MSA }                  from '../subworkflows/msa'
include { STRUCTURE_PREDICTION } from '../subworkflows/structure_prediction'
include { MULTIQC }              from '../modules/nf-core/multiqc/main'

include { paramsSummaryMap       } from 'plugin/nf-validation'
include { fromSamplesheet        } from 'plugin/nf-validation'
include { paramsSummaryMultiqc   } from '../subworkflows/local/utils_pipeline'
include { softwareVersionsToYAML } from '../subworkflows/local/utils_pipeline'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    TEST_STRUCTURE_PREDICTION - Test workflow for structure prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    This workflow tests the following steps:
    1. MSA generation (MMseqs2)
    2. Structure prediction (e.g., AlphaFold2, ColabFold, ESMFold, Protenix)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow TEST_STRUCTURE_PREDICTION {
    take: 
    ch_versions

    main:
    ch_multiqc_files = Channel.empty()

    //
    // Create input channel from samplesheet
    // Expected format: sample_id,fasta_file,a3m_file
    //
    Channel
        .fromPath(params.input)
        .splitCsv(header: true)
        .map { row ->
            def meta = [
                id: row.sample_id
            ]
            tuple(meta, file(row.fasta_file), file(row.a3m_file))
        }
        .set { ch_fasta_a3m }

    ch_fasta_a3m.view { "Input for structure prediction: $it" }

    //
    // STEP 1: MSA Generation (SKIPPED)
    //
    //
    // STEP 2: Structure Prediction
    //
    STRUCTURE_PREDICTION(ch_fasta_a3m)
    ch_versions = ch_versions.mix(STRUCTURE_PREDICTION.out.versions)
    ch_multiqc_files = ch_multiqc_files.mix(STRUCTURE_PREDICTION.out.multiqc.collect())
    
    //
    // Collate and save software versions
    //
    softwareVersionsToYAML(ch_versions)
        .collectFile(storeDir: "${params.outdir}/pipeline_info", name: 'test_structure_prediction_software_versions.yml', sort: true, newLine: true)
        .set { ch_collated_versions }

    //
    // MODULE: MultiQC
    //
    ch_multiqc_report = Channel.empty()
    if (!params.skip_multiqc) {
        ch_multiqc_config                     = Channel.fromPath("$projectDir/assets/multiqc_config.yml", checkIfExists: true)
        ch_multiqc_custom_config              = params.multiqc_config ? Channel.fromPath(params.multiqc_config) : Channel.empty()
        ch_multiqc_logo                       = params.multiqc_logo   ? Channel.fromPath(params.multiqc_logo)   : Channel.empty()
        summary_params                        = paramsSummaryMap(workflow, parameters_schema: "nextflow_schema.json")
        ch_workflow_summary                   = Channel.value(paramsSummaryMultiqc(summary_params))

        ch_multiqc_files = ch_multiqc_files.mix(ch_workflow_summary.collectFile(name: 'workflow_summary_mqc.yaml'))
        ch_multiqc_files = ch_multiqc_files.mix(ch_collated_versions)

        MULTIQC (
            ch_multiqc_files.collect(),
            ch_multiqc_config.toList(),
            ch_multiqc_custom_config.toList(),
            ch_multiqc_logo.toList(),
            Channel.empty().toList(),
            Channel.empty().toList()
        )
        ch_multiqc_report = MULTIQC.out.report.toList()
    }

    emit:
    predicted_pdbs = STRUCTURE_PREDICTION.out.pdb // channel: [meta, pdb]
    multiqc_report = ch_multiqc_report             // channel: /path/to/multiqc_report.html
    versions       = ch_versions                   // channel: [ path(versions.yml) ]
} 