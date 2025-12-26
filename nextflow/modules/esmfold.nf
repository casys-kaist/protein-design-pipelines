/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT LOCAL MODULES/SUBWORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

//
// MODULE: Loaded from modules/local/
//
include { MULTIFASTA_TO_SINGLEFASTA } from './local/multifasta_to_singlefasta'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
process RUN_ESMFOLD {
    tag "$meta.id"
    // Approximately 9880 MiB (seems to differ based on sequence length)
    accelerator 11, type: 'aliyun.com/gpu-mem'
    // accelerator 1, type: 'aliyun.com/gpu-count'
    // maxForks 1
    accelerator 1, type: 'aliyun.com/gpu-count'
    // maxForks 1

    label 'gpu'

    
    container "quay.io/nf-core/proteinfold_esmfold:1.1.1"

    input:
    tuple val(meta), path(fasta)
    path ('./checkpoints/')
    val(num_cycles)

    output:
    tuple val(meta), path ("${fasta.baseName}*.pdb"), emit: pdb
    path ("${fasta.baseName}_plddt_mqc.tsv"), emit: multiqc
    path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def VERSION = '1.0.3' // WARN: Version information not provided by tool on CLI. Please update this string when bumping container versions.

    """
    esm-fold \
        -i ${fasta} \
        -o \$PWD \
        -m \$PWD \
        --num-recycles ${num_cycles} \
        $args

    awk '{print \$2"\\t"\$3"\\t"\$4"\\t"\$6"\\t"\$11}' "${fasta.baseName}"*.pdb | grep -v 'N/A' | uniq > plddt.tsv
    echo -e Atom_serial_number"\\t"Atom_name"\\t"Residue_name"\\t"Residue_sequence_number"\\t"pLDDT > header.tsv
    cat header.tsv plddt.tsv > "${fasta.baseName}"_plddt_mqc.tsv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        esm-fold: $VERSION
END_VERSIONS
    """

    stub:
    def VERSION = '1.0.3' // WARN: Version information not provided by tool on CLI. Please update this string when bumping container versions.
    """
    touch ./"${fasta.baseName}".pdb
    touch ./"${fasta.baseName}"_plddt_mqc.tsv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        esm-fold: $VERSION
END_VERSIONS
    """
}

workflow ESMFOLD {

    take:
    ch_versions       // channel: [ path(versions.yml) ]
    ch_fasta          // channel: [ path(fasta) ]
    ch_esmfold_params // directory: /path/to/esmfold/params/
    ch_num_cycles   // int: Number of recycles for esmfold

    main:
    ch_multiqc_files = Channel.empty()
    ch_pdb = Channel.empty()

    //
    // MODULE: Run esmfold
    //
    if (params.esmfold_model_preset != 'monomer') {
        MULTIFASTA_TO_SINGLEFASTA(
            ch_fasta
        )
        ch_versions = ch_versions.mix(MULTIFASTA_TO_SINGLEFASTA.out.versions)
        RUN_ESMFOLD(
            MULTIFASTA_TO_SINGLEFASTA.out.input_fasta,
            ch_esmfold_params,
            params.esmfold_num_cycles
        )
        ch_versions = ch_versions.mix(RUN_ESMFOLD.out.versions)
        ch_pdb = RUN_ESMFOLD.out.pdb
    } else {
        RUN_ESMFOLD(
            ch_fasta,
            ch_esmfold_params,
            params.esmfold_num_cycles
        )
        ch_versions = ch_versions.mix(RUN_ESMFOLD.out.versions)
        ch_multiqc_files.mix(RUN_ESMFOLD.out.multiqc.collect())
        ch_pdb = RUN_ESMFOLD.out.pdb.transpose()
    }


    emit:
    multiqc        = ch_multiqc_files
    versions       = ch_versions       // channel: [ path(versions.yml) ]
    pdb            = ch_pdb
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
