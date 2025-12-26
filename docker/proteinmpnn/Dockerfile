# Copied from https://github.com/CyrusBiotechnology/ProteinMPNN-docker
FROM nvidia/cuda:11.8.0-base-ubuntu22.04
# COPY . /home/ProteinMPNN

# Install base utilities
RUN apt-get update \
    && apt-get install -y \
        build-essential \
        wget \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install biopython
RUN pip install 'numpy<2.0'
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /workspace