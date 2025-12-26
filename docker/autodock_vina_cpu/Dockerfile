FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

# Basic system tools and dependencies
RUN apt-get update && apt-get install -y wget

RUN wget https://micro.mamba.pm/api/micromamba/linux-64/latest -O micromamba.tar.bz2 \
    && mkdir -p /opt/micromamba \
    && tar -xvjf micromamba.tar.bz2 -C /opt/micromamba \
    && mv /opt/micromamba/bin/micromamba /usr/local/bin/micromamba \
    && rm -rf micromamba.tar.bz2 /opt/micromamba

RUN micromamba install -c conda-forge -c bioconda "vina>=1.2.3" openbabel meeko mdanalysis mgltools \
    && micromamba clean --all --yes

WORKDIR /workspace

CMD ["micromamba", "run", "-n", "base", "bash"]