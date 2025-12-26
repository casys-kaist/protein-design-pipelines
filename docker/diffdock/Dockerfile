FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

RUN apt-get update -y && apt-get install -y gcc wget curl git tar bzip2 unzip && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /opt && \
    mkdir -p /opt/conda/bin && \
    mv /opt/bin/micromamba /opt/conda/bin/

ENV CONDA_DEFAULT_ENV=diffdock

COPY environment.yml /tmp/environment.yml

RUN micromamba create -y -n diffdock -f /tmp/environment.yml && \
    rm /tmp/environment.yml

SHELL ["micromamba", "run", "-n", "diffdock", "bin/bash", "-c"]

CMD ["bash"]
