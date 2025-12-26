nextflow.enable.dsl=2

include { STRUCTURE_DESIGN }     from '../subworkflows/structure_design'
// include { REFINEMENT }         from '../subworkflows/refinement'
include { INVERSE_FOLDING }      from '../subworkflows/inverse_folding'
include { VARIANT_GEN }          from '../subworkflows/variant_generation'
include { MSA }                  from '../subworkflows/msa'
include { STRUCTURE_PREDICTION } from '../subworkflows/structure_prediction'
include { DOCK_LIGAND }          from '../subworkflows/docking'
// include { SCORE_BINDING }     from '../subworkflows/scoring'

include { MULTIQC } from '../modules/nf-core/multiqc/main'

include { paramsSummaryMap       } from 'plugin/nf-validation'
include { fromSamplesheet        } from 'plugin/nf-validation'
include { paramsSummaryMultiqc   } from '../subworkflows/local/utils_pipeline'
include { softwareVersionsToYAML } from '../subworkflows/local/utils_pipeline'

workflow DENOVO_LIGAND {
    take: 
    ch_versions

    main:
    ch_multiqc_files = Channel.empty()

    //
    // Create input channel from input file provided through params.input
    //
    Channel
        .fromSamplesheet("input")
        .map { row ->
            // Expected columns: sequence, ligand_sdf, ligand_pdbqt, docked_pdb, ligand_mol2, contigs, exhaustiveness
            def meta = [
                id           : row[0],
                ligand_sdf   : file(row[1]),
                ligand_pdbqt : file(row[2]),
                ligand_mol2  : file(row[4]),
                contigs      : row[5],
            ]
            tuple ( meta, file(row[3]))
        }
        .set { ch_pdb }
    if (params.debug_channels) {
        ch_pdb.view { it -> "PDB_IN: ${it[0].id} -> ${it[1].name}" }
    }

    // Initialize channels for conditional execution
    ch_design_pdb = ch_pdb
    ch_folding_fasta = Channel.empty()
    ch_variant_fasta = Channel.empty()
    ch_msa_fasta = Channel.empty()
    ch_prediction_pdb = Channel.empty()

    //
    // STEP 1: Structure Design (conditional)
    //
    if (params.structure_design_mode && params.structure_design_mode != 'null') {
        STRUCTURE_DESIGN( ch_pdb )
        ch_design_pdb = STRUCTURE_DESIGN.out.pdb
        ch_versions = ch_versions.mix(STRUCTURE_DESIGN.out.versions)
    }

    //
    // STEP 2: Inverse Folding (conditional)
    //
    if (params.inverse_folding_mode && params.inverse_folding_mode != 'null') {
        INVERSE_FOLDING( ch_design_pdb )
        ch_folding_fasta = INVERSE_FOLDING.out.fasta
        ch_versions = ch_versions.mix(INVERSE_FOLDING.out.versions)
        if (params.debug_channels) {
            ch_folding_fasta.view { it -> "FOLD_OUT: ${it[0].id} -> ${it[1].name}" }
        }
    }

    //
    // STEP 3: Variant Generation (ESM-2)
    //
    if (params.variant_generation_mode && params.variant_generation_mode != 'null') {
        VARIANT_GEN( ch_folding_fasta )
        ch_variant_fasta = VARIANT_GEN.out.fasta
        ch_versions = ch_versions.mix(VARIANT_GEN.out.versions)
        if (params.debug_channels) {
            ch_variant_fasta.view { it -> "VARIANT_OUT: ${it[0].id} -> ${it[1].name}" }
        }
    } else {
        ch_variant_fasta = ch_folding_fasta
    }

    //
    // STEP 4: MSA (conditional - required for Protenix; skipped for ESMFold)
    //
    if (params.msa_mode && params.msa_mode != 'null' &&
        params.structure_prediction_mode != 'esmfold' &&
        params.structure_prediction_mode != 'null') {
        MSA( ch_variant_fasta )
        ch_fasta_a3m = MSA.out.fasta_a3m
        ch_versions = ch_versions.mix(MSA.out.versions)
        if (params.debug_channels) {
            ch_fasta_a3m.view { it -> "MSA_OUT_MAIN: ${it[0].id} -> ${it[2].name}" }
        }
    } else {
        // ESMFold will skip MSA, so we need to pass the fasta to structure prediction
        ch_fasta_a3m = ch_variant_fasta
    }

    //
    // STEP 5: Structure Prediction (conditional)
    //
    if (params.structure_prediction_mode && params.structure_prediction_mode != 'null') {
        STRUCTURE_PREDICTION( ch_fasta_a3m )
        ch_prediction_pdb = STRUCTURE_PREDICTION.out.pdb
        ch_versions = ch_versions.mix(STRUCTURE_PREDICTION.out.versions)
        if (params.debug_channels) {
            ch_prediction_pdb.view { it -> "PRED_OUT_MAIN: ${it[0].id} -> ${it[1].name}" }
        }
    } else {
        ch_prediction_pdb = ch_design_pdb
    }

    //
    // STEP 6: Docking (conditional)
    //
    if (params.docking_mode && params.docking_mode != 'null') {
        DOCK_LIGAND( ch_prediction_pdb )
        ch_docked_protein_ligand = DOCK_LIGAND.out.protein_ligand
        ch_versions = ch_versions.mix(DOCK_LIGAND.out.versions)
        if (params.debug_channels) {
            ch_docked_protein_ligand.view { it -> "DOCK_OUT: ${it[0].id} -> ${it[1].name}" }
        }
    }

    //
    // Collate and save software versions
    // (Versions are already collected above in conditional blocks)
    //
    softwareVersionsToYAML(ch_versions)
        .collectFile(storeDir: "${params.outdir}/pipeline_info", name: 'nf_core_proteinfold_software_mqc_versions.yml', sort: true, newLine: true)
        .set { ch_collated_versions }

    //
    // MODULE: MultiQC
    //
    ch_multiqc_report = Channel.empty()
    if (!params.skip_multiqc) {
        ch_multiqc_report                     = Channel.empty()
        ch_multiqc_config                     = Channel.fromPath("$projectDir/assets/multiqc_config.yml", checkIfExists: true)
        ch_multiqc_custom_config              = params.multiqc_config ? Channel.fromPath( params.multiqc_config ) : Channel.empty()
        ch_multiqc_logo                       = params.multiqc_logo   ? Channel.fromPath( params.multiqc_logo )   : Channel.empty()
        summary_params                        = paramsSummaryMap(workflow, parameters_schema: "nextflow_schema.json")
        ch_workflow_summary                   = Channel.value(paramsSummaryMultiqc(summary_params))

        ch_multiqc_files = Channel.empty()
        ch_multiqc_files = ch_multiqc_files.mix(ch_workflow_summary.collectFile(name: 'workflow_summary_mqc.yaml'))
        ch_multiqc_files = ch_multiqc_files.mix(ch_collated_versions)
        
        // Conditionally collect MultiQC files based on executed steps
        if (params.inverse_folding_mode && params.inverse_folding_mode != 'null') {
            ch_multiqc_files = ch_multiqc_files.mix(INVERSE_FOLDING.out.multiqc.collect())
        }
        if (params.msa_mode && params.msa_mode != 'null' && 
            params.structure_prediction_mode != 'esmfold' && 
            params.structure_prediction_mode != 'null') {
            ch_multiqc_files = ch_multiqc_files.mix(MSA.out.multiqc.collect())
        }
        if (params.structure_prediction_mode && params.structure_prediction_mode != 'null') {
            ch_multiqc_files = ch_multiqc_files.mix(STRUCTURE_PREDICTION.out.multiqc.collect())
        }

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
    multiqc_report = ch_multiqc_report // channel: /path/to/multiqc_report.html
    versions       = ch_versions       // channel: [ path(versions.yml) ]
}
