nextflow.enable.dsl=2

include { STRUCTURE_DESIGN } from '../subworkflows/structure_design'
include { MULTIQC }     from '../modules/nf-core/multiqc/main'

include { paramsSummaryMap       } from 'plugin/nf-validation'
include { paramsSummaryMultiqc   } from '../subworkflows/local/utils_pipeline'
include { softwareVersionsToYAML } from '../subworkflows/local/utils_pipeline'

workflow TEST_STRUCTURE_DESIGN {
    take:
    ch_versions

    main:
    ch_multiqc_files = Channel.empty()

    Channel
        .fromPath(params.input)
        .splitCsv(header: true)
        .map { row ->
            def meta = [
                id: row.sample_id,
                contigs: row.contigs
            ]
            tuple(meta, file(row.pdb_file))
        }
        .set { ch_pdb }
    
    ch_pdb.view { "Input PDB: $it" }

    STRUCTURE_DESIGN(ch_pdb)
    softwareVersionsToYAML(ch_versions)
        .collectFile(storeDir: "${params.outdir}/pipeline_info", name: 'test_docking_software_versions.yml', sort: true, newLine: true)
        .set { ch_collated_versions }

    //
    // MODULE: MultiQC
    //
    ch_multiqc_report = Channel.empty()
    if (!params.skip_multiqc) {
        ch_multiqc_config                     = Channel.fromPath("$projectDir/assets/multiqc_config.yml", checkIfExists: true)
        ch_multiqc_custom_config              = params.multiqc_config ? Channel.fromPath( params.multiqc_config ) : Channel.empty()
        ch_multiqc_logo                       = params.multiqc_logo   ? Channel.fromPath( params.multiqc_logo )   : Channel.empty()
        summary_params                        = paramsSummaryMap(workflow, parameters_schema: "nextflow_schema.json")
        ch_workflow_summary                   = Channel.value(paramsSummaryMultiqc(summary_params))

        ch_multiqc_files = Channel.empty()
        ch_multiqc_files = ch_multiqc_files.mix(ch_workflow_summary.collectFile(name: 'workflow_summary_mqc.yaml'))
        ch_multiqc_files = ch_multiqc_files.mix(ch_collated_versions)

        MULTIQC (
            ch_multiqc_files.collect(),
            ch_multiqc_config.toList(),
            ch_multiqc_custom_config.toList(),
            ch_multiqc_logo.toList(),
            Channel.empty().toList(),
            Channel.empty().toList(),
        )
        ch_multiqc_report = MULTIQC.out.report.toList()
    }

    emit:
    multiqc_report = ch_multiqc_report        // channel: /path/to/multiqc_report.html
    versions       = ch_versions           
}

