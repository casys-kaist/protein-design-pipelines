nextflow.enable.dsl = 2

include { DENOVO_LIGAND }   from './workflows/denovo_design_ligand'
include { TEST_DOCKING }    from './workflows/test_docking'
include { TEST_MSA }        from './workflows/test_msa'
include { TEST_STRUCTURE_PREDICTION } from './workflows/test_structure_prediction'
include { TEST_STRUCTURE_DESIGN } from './workflows/test_structure_design'
// include { PROTAC_DESIGN }          from './workflows/protac_design'
// include { DENOVO_DESIGN_PROTEIN }  from './workflows/denovo_design_protein'

/* Default: denovo_design_ligand */
params.pipeline = params.pipeline ?: 'denovo_design_ligand'


workflow MAIN {
    main:
    ch_multiqc  = Channel.empty()
    ch_versions = Channel.empty()

    if (params.pipeline == 'denovo_design_ligand') {
        DENOVO_LIGAND( ch_versions )
        ch_multiqc  = DENOVO_LIGAND.out.multiqc_report
        ch_versions = ch_versions.mix(DENOVO_LIGAND.out.versions)
    } else if (params.pipeline == 'test_docking') {
        TEST_DOCKING( ch_versions )
        ch_multiqc  = TEST_DOCKING.out.multiqc_report
        ch_versions = ch_versions.mix(TEST_DOCKING.out.versions)
    } else if (params.pipeline == 'test_msa') {
        TEST_MSA( ch_versions )
        ch_multiqc  = TEST_MSA.out.multiqc_report
        ch_versions = ch_versions.mix(TEST_MSA.out.versions)
    } else if (params.pipeline == 'test_structure_prediction') {
        TEST_STRUCTURE_PREDICTION( ch_versions )
        ch_multiqc = TEST_STRUCTURE_PREDICTION.out.multiqc_report
        ch_versions = ch_versions.mix(TEST_STRUCTURE_PREDICTION.out.versions)
    } else if (params.pipeline == 'test_structure_design') {
        TEST_STRUCTURE_DESIGN( ch_versions )
        ch_multiqc = TEST_STRUCTURE_DESIGN.out.multiqc_report
        ch_versions = ch_versions.mix(TEST_STRUCTURE_DESIGN.out.versions)
    } else {
        throw new Exception("Pipeline ${params.pipeline} not supported")
    }

    emit:
    multiqc_report = ch_multiqc  // channel: /path/to/multiqc_report.html
    versions       = ch_versions // channel: [version1, version2, ...]
}

workflow {
    main:
    MAIN ()
}
