#!/usr/bin/env nextflow
nextflow.enable.dsl = 2
include { CONVERT_PDB_TO_PDBQT } from '../modules/vina_gpu'
include { VINA_CPU } from '../modules/vina_cpu'
include { VINA_GPU } from '../modules/vina_gpu'
include { DIFFDOCK } from '../modules/diffdock'


/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    DOCK_LIGAND
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
workflow DOCK_LIGAND {
    take:
        ch_pdb

    main:
    ch_versions = Channel.empty()
    ch_result = Channel.empty()
    if (params.docking_mode.contains('vina')) {
        // Prepare pdbqt files
        CONVERT_PDB_TO_PDBQT(
            ch_pdb,
        )
        ch_versions = ch_versions.mix(CONVERT_PDB_TO_PDBQT.out.versions)
        CONVERT_PDB_TO_PDBQT.out.pdbqt.map(
            {
                meta, pdbqt, center_txt ->
                def props = center_txt.text.readLines().collectEntries {
                    def parts = it.split('=')
                    [(parts[0]): parts[1]]
                }
                meta.center_x = props.center_x
                meta.center_y = props.center_y
                meta.center_z = props.center_z
                def ligand_pdbqt_file = file(meta.ligand_pdbqt)
                [meta, pdbqt, ligand_pdbqt_file]
            }
        ).set {
            ch_pdbqt
        }

        if (params.docking_mode == 'vina_cpu') {
            VINA_CPU(ch_pdbqt)
            ch_result = VINA_CPU.out.pdb
            ch_versions = ch_versions.mix(VINA_CPU.out.versions)
        } else if (params.docking_mode == 'vina_gpu') {
            VINA_GPU(ch_pdbqt)
            ch_result = VINA_GPU.out.pdb
            ch_versions = ch_versions.mix(VINA_GPU.out.versions)
        } else {
            throw new Exception("Docking mode not supported: ${params.docking_mode}")
        }

    } else if (params.docking_mode == "diffdock") {
        DIFFDOCK(ch_pdb)
        ch_result = DIFFDOCK.out.protein_ligand
        ch_versions = ch_versions.mix(DIFFDOCK.out.versions)
    } else {
        throw new Exception("Docking mode not supported: ${params.docking_mode}")
    }

    emit:
    protein_ligand      = ch_result
    versions = ch_versions
}
