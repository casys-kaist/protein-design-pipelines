// Utility helpers for this pipeline (MultiQC summary + version collation).

// Get software versions for pipeline
// Input: channel of versions.yml files
// Output: channel of YAML strings

def processVersionsFromYAML(yaml_file) {
    def yaml = new org.yaml.snakeyaml.Yaml()
    def versions = yaml.load(yaml_file).collectEntries { k, v -> [k.tokenize(':')[-1], v] }
    return yaml.dumpAsMap(versions).trim()
}

def workflowVersionToYAML() {
    def name = workflow.manifest.name ?: 'pipeline'
    def version_string = ''
    if (workflow.manifest.version) {
        def prefix_v = workflow.manifest.version[0] != 'v' ? 'v' : ''
        version_string += "${prefix_v}${workflow.manifest.version}"
    }
    if (workflow.commitId) {
        def git_shortsha = workflow.commitId.substring(0, 7)
        version_string += "-g${git_shortsha}"
    }
    if (!version_string) {
        version_string = 'unknown'
    }

    return """
    Workflow:
        ${name}: ${version_string}
        Nextflow: ${workflow.nextflow.version}
    """.stripIndent().trim()
}

// Get channel of software versions used in pipeline in YAML format

def softwareVersionsToYAML(ch_versions) {
    return ch_versions.unique().map { version -> processVersionsFromYAML(version) }.unique().mix(Channel.of(workflowVersionToYAML()))
}

// Get workflow summary for MultiQC

def paramsSummaryMultiqc(summary_params) {
    def summary_section = ''
    summary_params
        .keySet()
        .each { group ->
            def group_params = summary_params.get(group)
            if (group_params) {
                summary_section += "    <p style=\"font-size:110%\"><b>${group}</b></p>\n"
                summary_section += "    <dl class=\"dl-horizontal\">\n"
                group_params
                    .keySet()
                    .sort()
                    .each { param ->
                        summary_section += "        <dt>${param}</dt><dd><samp>${group_params.get(param) ?: '<span style=\\\"color:#999999;\\\">N/A</a>'}</samp></dd>\n"
                    }
                summary_section += "    </dl>\n"
            }
        }

    def name = workflow.manifest.name ?: 'pipeline'
    def href = workflow.manifest.homePage ?: ''

    def yaml_file_text = "id: '${name.replace('/', '-')}-summary'\n" as String
    yaml_file_text     += "description: 'Pipeline parameters captured at launch.'\n"
    yaml_file_text     += "section_name: '${name} Workflow Summary'\n"
    if (href) {
        yaml_file_text += "section_href: '${href}'\n"
    }
    yaml_file_text     += "plot_type: 'html'\n"
    yaml_file_text     += "data: |\n"
    yaml_file_text     += "${summary_section}"

    return yaml_file_text
}
