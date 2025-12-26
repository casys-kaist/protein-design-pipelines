/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
process RUN_ESM {
    tag "$meta.id"

    // GPU accelerator sizing is configured via Nextflow config (see RUN_ESM selector)
    accelerator 12, type: 'aliyun.com/gpu-mem'
    memory 15.GB

    label 'gpu'

    container "protenix:latest"

    input:
    tuple val(meta), path(fasta)
    val num_variants
    val max_mutations
    val top_k   

    output:
    tuple val(meta), path ("variants.json"), emit: variant_json
    tuple val(meta), path ("*_esm_*.fasta"), emit: fasta
    path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def modelName = params.esm_model_name ?: 'facebook/esm2_t33_650M_UR50D'
    def extraArgs = task.ext.args ? task.ext.args.trim() : ''
    def cacheDir = params.esm_cache_dir ?: ''
    def cacheArg = cacheDir ? "--cache_dir ${cacheDir}" : ''
    def cliArgs = ["--model_name ${modelName}", cacheArg, extraArgs].findAll { it }.join(' ')

    """
    CURRENT_DIR=\$(pwd)

    (cd /workspace/esm && python generate_variants.py \
        --ref_fasta_file \$CURRENT_DIR/${fasta} \
        --output_dir \$CURRENT_DIR \
        --num_variants ${num_variants} \
        --max_mutations ${max_mutations} \
        --top_k ${top_k} \
        ${cliArgs})

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        esm: ${modelName}
END_VERSIONS
    """

    stub:
    """
    touch ./variants.json
    echo ">stub_sample_esm_0\nSEQUENCE" > stub_sample_esm_0.fasta
    echo ">stub_sample_esm_1\nSEQUENCE" > stub_sample_esm_1.fasta

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        esm: ${params.esm_model_name ?: 'facebook/esm2_t33_650M_UR50D'}
END_VERSIONS
    """
}

workflow ESM {

    take:
    ch_fasta          // channel: [ val(meta), path(fasta) ]
    ch_num_variants   // int: Number of variants generated
    ch_max_mutations   // int: Number of mutations
    ch_top_k           // int: Sample from the top-k predictions for each masked position

    main:
    //
    // MODULE: Run esm
    //

    RUN_ESM(
        ch_fasta,
        ch_num_variants,
        ch_max_mutations,
        ch_top_k
    )
    ch_multiqc_files = Channel.empty()

    emit:
    versions     = RUN_ESM.out.versions
    variant_json = RUN_ESM.out.variant_json
    fasta        = RUN_ESM.out.fasta.transpose().map { meta, fasta ->
        def variantMeta = meta.clone()
        def baseName = fasta.baseName
        variantMeta.put('parent_id', meta.id)
        variantMeta.id = baseName
        tuple(variantMeta, fasta)
    }
    multiqc      = ch_multiqc_files
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
