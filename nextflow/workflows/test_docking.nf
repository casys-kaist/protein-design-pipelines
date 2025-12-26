#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { DOCK_LIGAND } from '../subworkflows/docking'
include { MULTIQC }     from '../modules/nf-core/multiqc/main'

include { paramsSummaryMap       } from 'plugin/nf-validation'
include { fromSamplesheet        } from 'plugin/nf-validation'
include { paramsSummaryMultiqc   } from '../subworkflows/local/utils_pipeline'
include { softwareVersionsToYAML } from '../subworkflows/local/utils_pipeline'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    TEST_DOCKING - Simple test workflow for docking only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow TEST_DOCKING {
    take: 
    ch_versions

    main:
    ch_multiqc_files = Channel.empty()

    //
    // Create input channel from input file provided through params.input
    // Expected format: sample_id, ligand_sdf, ligand_pdbqt, scaffold_pdb, ligand_mol2
    // - scaffold_pdb can also be provided as receptor_pdb (backwards compatible)
    // - ligand_pdbqt can also be provided as docked_pdbqt (backwards compatible)
    // Skip validation for test_docking workflow
    //
    Channel
        .fromPath(params.input)
        .splitCsv(header: true)
        .map { row ->
            def scaffoldPdb = row.scaffold_pdb ?: row.receptor_pdb
            def ligandPdbqt = row.ligand_pdbqt ?: row.docked_pdbqt

            if (!row.sample_id || !row.ligand_sdf || !row.ligand_mol2 || !scaffoldPdb) {
                throw new IllegalArgumentException("Samplesheet row is missing required columns (need sample_id, ligand_sdf, scaffold_pdb/receptor_pdb, ligand_mol2): ${row}")
            }
            if (params.docking_mode?.contains('vina') && !ligandPdbqt) {
                throw new IllegalArgumentException("Vina docking requires ligand_pdbqt (or docked_pdbqt) column for sample ${row.sample_id}")
            }

            def meta = [
                id           : row.sample_id,
                ligand_sdf   : file(row.ligand_sdf),
                ligand_pdbqt : ligandPdbqt ? file(ligandPdbqt) : null,
                ligand_mol2  : file(row.ligand_mol2)
            ]
            tuple ( meta, file(scaffoldPdb))
        }
        .set { ch_pdb }
    
    ch_pdb.view { "Input PDB: $it" }

    // Run docking directly on provided PDB files
    DOCK_LIGAND( ch_pdb )

    //
    // Collate and save software versions
    //
    ch_versions = ch_versions.mix(
        DOCK_LIGAND.out.versions
    )
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
    docked_pdbs    = DOCK_LIGAND.out.protein_ligand      // channel: [meta, pdb, ligand_path]
    multiqc_report = ch_multiqc_report        // channel: /path/to/multiqc_report.html
    versions       = ch_versions              // channel: [ path(versions.yml) ]
} 
