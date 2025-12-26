# FROM ghcr.io/soedinglab/mmseqs2:17-b804f-cuda12
FROM ghcr.io/soedinglab/mmseqs2:master-cuda12

RUN apt update \
    && apt install -y \
        git \
        vim \
        wget \
        bzip2 \
        tmux \
        htop \
        procps \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

ENV PATH /opt/conda/bin:$PATH

RUN git config --global --add safe.directory /workspace

RUN ln /usr/local/bin/mmseqs_avx2 /usr/local/bin/mmseqs


RUN pip install --no-cache-dir absl-py
RUN pip install --no-cache-dir importlib-metadata
RUN pip install --no-cache-dir biopython

ARG COLABFOLD_REF=e6f3ddd10580a83d706a8cf6ba3ac928ea3ce2bc
RUN pip install --no-cache-dir --no-deps "git+https://github.com/jinuhwang/Colabfold.git@${COLABFOLD_REF}"

ENTRYPOINT []
