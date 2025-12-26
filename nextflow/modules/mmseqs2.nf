/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    MMseqs2 MODULE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

process RUN_MMSEQS2 {
    container "colabfold:latest"
    
    // Seems to require max GPU memory
    accelerator 24, type: 'aliyun.com/gpu-mem'
    label 'gpu'
    memory 46.GB
    // accelerator 1, type: 'aliyun.com/gpu-count'

    // maxForks 1  
    
    input:
    tuple val(meta_list), path(fasta_list)
    val colabfold_db
    val db1
    val db3
    val num_iterations

    output:
    tuple val(meta_list), path("out/*.a3m"), emit: a3m
    path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def VERSION = '14.7e284' // MMseqs2 version

    """
    PWD=\$(pwd)

    # 1) concat N fasta files
    cat ${fasta_list} > batch.fasta

    mkdir -p out

    # 2) run Colabfold search in batch
    # colabfold_search \\
    #     batch.fasta \\
    #     ${colabfold_db} \\
    #     \$PWD/out/ \\
    #     --db1 ${db1} \\
    #     --db3 ${db3} \\
    #     --use-env 1 \\
    #     --db-load-mode 2 \\
    #     --prefilter-mode 1 \\
    #     --gpu 1 \\
    #     $args

    # excluded env db to avoid disk contention
    colabfold_search \\
        batch.fasta \\
        ${colabfold_db} \\
        \$PWD/out/ \\
        --db1 ${db1} \\
        --db-load-mode 2 \\
        --prefilter-mode 1 \\
        --gpu 1 \\
        --use-env 0 \\
        $args

    #  #  --num-iterations ${num_iterations} \\
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        mmseqs2: $VERSION
END_VERSIONS
    """

    stub:
    def VERSION = '14.7e284'
    def stub_commands = meta_list
        .collect { meta ->
            def candidate = meta.mmseqs2_output_a3m ?: meta.sample_id ?: meta.id ?: 'stub_sample'
            candidate = candidate.toString()
            if (!candidate.endsWith('.a3m')) {
                candidate = candidate + '.a3m'
            }
            "touch out/${candidate}"
        }
        .unique()
        .join('\n')
    if (!stub_commands) {
        stub_commands = 'touch out/stub_sample.a3m'
    }
    """
    mkdir -p out
    ${stub_commands}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        mmseqs2: $VERSION
END_VERSIONS
    """
}

workflow MMSEQS2 {
    take:
    ch_versions       // channel: [ path(versions.yml) ]
    ch_fasta          // channel: [ meta, path(fasta) ]
    val_colabfold_db  // val: path(colabfold_db)
    val_db1           // val: db1 parameter
    val_db3           // val: db3 parameter
    val_batch_size    // val: batch size
    val_num_iterations // val: number of search iterations

    main:


    // First, mark input file names to unbatch after running mmseqs2
    ch_fasta_batched = ch_fasta.map(
                           { meta, fasta ->
                               def m = meta.clone()
                               m.mmseqs2_input_fasta = fasta
                               // carry a stable sample id for routing outputs
                               m.sample_id = meta.id
                               m.mmseqs2_output_a3m = fasta.name.replaceAll('\\.fasta$', '.a3m')
                               tuple(m, fasta)
                           }
                       )
                       .collate(val_batch_size)
                       .map { batch ->
                           tuple( batch.collect { it[0] }, batch.collect { it[1] })
                       }

    //
    // MODULE: Run MMseqs2 (GPU/CPU accelerator set by config)
    //
    RUN_MMSEQS2(
        ch_fasta_batched,
        val_colabfold_db,
        val_db1,
        val_db3,
        val_num_iterations
    )

    // Second, unbatch the output
    ch_a3m_unbatched = RUN_MMSEQS2.out.a3m.flatMap { meta_list, a3m_files ->
        // ensure we always have a List<Path>
        List<Path> a3m_list = (a3m_files instanceof List) ? a3m_files : [a3m_files]

        a3m_list.collect { a3m ->
            // Route by prefix: '<sample_id>_esm_*' produced from FASTA headers
            def base = a3m.getBaseName()
            def meta = meta_list.find { it.sample_id == base }
            if (!meta && base.contains('_esm_')) {
                def samplePrefix = base.split('_esm_')[0]
                meta = meta_list.find { it.sample_id == samplePrefix }
            }
            if (!meta && meta_list.size() == 1) {
                meta = meta_list[0]
            }
            if( !meta ) error "No meta entry for output ${a3m.name} – got sample_ids: ${meta_list*.sample_id}"
            tuple(meta, meta.mmseqs2_input_fasta as Path, a3m)
        }
    }

    ch_versions = ch_versions.mix(RUN_MMSEQS2.out.versions)

    emit:
    fasta_a3m      = ch_a3m_unbatched
    versions       = ch_versions       // channel: [ path(versions.yml) ]
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/ 
