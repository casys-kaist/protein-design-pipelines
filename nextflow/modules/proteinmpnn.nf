/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
process RUN_PROTEINMPNN {
    tag "$meta.id"
    // Approximately 430 MiB
    accelerator 1, type: 'aliyun.com/gpu-mem'
    // accelerator 1, type: 'aliyun.com/gpu-count'
    // maxForks 1

    label 'gpu'
    memory 1.GB

    
    container "proteinmpnn:latest"

    input:
    tuple val(meta), path(pdb)
    val num_seq_per_target
    val sampling_temp

    output:
    // Per-sequence FASTAs only; downstream will consume one sequence per job
    tuple val(meta), path ("${pdb.baseName}_mpnn_*.fasta"), emit: fasta
    path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''

    """
    CURRENT_DIR=\$(pwd)
    echo "NF process working directory: \$CURRENT_DIR"
    echo "Contents of CURRENT_DIR:"
    ls -lh
    echo "PDB path (Nextflow expanded) = ${pdb}"
    echo "PDB basename = ${pdb.baseName}"
    (cd /workspace/third_parties/ProteinMPNN/examples && bash run_single_pdb.sh \
        -p \$CURRENT_DIR/${pdb} \
        -o \$CURRENT_DIR/${pdb.baseName} \
        -n ${num_seq_per_target} \
        -t ${sampling_temp} \
        $args)
    echo "hello"
    # Remove first sequence (native) and keep only generated sequences into a single multi-FASTA
    awk -v prefix="${pdb.baseName}_mpnn_" '
    /^>/ {
        if (++n > 1) {  # Skip first sequence (native)
            print ">" prefix (n-2);
            next
        } else {
            next  # Skip first header (native)
        }
    }
    n > 1 {print}  # Only print sequences after first one
    ' ${pdb.baseName}/seqs/*.fa > ${pdb.baseName}.fasta

    # Split multi-FASTA into one FASTA per sequence for downstream single-sequence processing
    # Output files: ${pdb.baseName}_mpnn_0.fasta, ${pdb.baseName}_mpnn_1.fasta, ...
    awk -v stem="${pdb.baseName}_mpnn_" '
      BEGIN{idx=-1}
      /^>/{
        if (out){ close(out) }
        idx++
        fn=sprintf("%s%d.fasta", stem, idx)
        out=fn
        print \$0 > out
        next
      }
      { print \$0 > out }
      END{ if (out){ close(out) } }
    ' ${pdb.baseName}.fasta

    # Optionally remove combined multi-FASTA if not needed further
    rm -f ${pdb.baseName}.fasta

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        proteinmpnn: null
END_VERSIONS
    """

    stub:
    """
    touch "${pdb.baseName}_mpnn_0.fasta"

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        proteinmpnn: null
END_VERSIONS
    """
}

workflow PROTEINMPNN {

    take:
    ch_pdb          // channel: [ path(pdb) ]
    ch_num_seq_per_target   // int: Number of recycles for esmfold
    ch_sampling_temp   // int: Number of recycles for esmfold

    main:
    //
    // MODULE: Run proteinmpnn
    //

    RUN_PROTEINMPNN(
        ch_pdb,
        ch_num_seq_per_target,
        ch_sampling_temp
    )
    // ch_multiqc_files.mix(RUN_PROTEINMPNN.out.multiqc.collect())
    ch_multiqc_files = Channel.empty()

    emit:
    versions       = RUN_PROTEINMPNN.out.versions
    fasta          = RUN_PROTEINMPNN.out.fasta.transpose().map { meta, fasta ->
        def newMeta = meta.clone()
        newMeta.put('parent_id', meta.id)
        newMeta.id = fasta.baseName
        newMeta.sample_label = newMeta.id
        tuple(newMeta, fasta)
    }
    multiqc        = ch_multiqc_files
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
