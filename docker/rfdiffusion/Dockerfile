# Built from third_parties/RFdiffusion/docker/Dockerfile
# FROM rfdiffusion:latest
# Adapted from https://github.com/jinwooh/rfdiffusion/blob/main/docker/Dockerfile
FROM nvcr.io/nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

RUN apt-get -q update \
  && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
  git \
  libcap2 \
  python3.9 \
  python3-pip \
  && python3.9 -m pip install -q -U --no-cache-dir pip \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get autoremove -y \
  && apt-get clean \
  && pip install -q --no-cache-dir \
  dgl==1.0.2+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html \
  torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 \
  e3nn==0.3.3 \
  wandb==0.12.0 \
  pynvml==11.0.0 \
  git+https://github.com/NVIDIA/dllogger#egg=dllogger \
  decorator==5.1.0 \
  hydra-core==1.3.2 \
  pyrsistent==0.19.3

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN python -m pip install 'numpy<2.0'

ADD RFdiffusion /workspace/third_parties/RFdiffusion
RUN python -m pip install -e /workspace/third_parties/RFdiffusion/env/SE3Transformer/ 
RUN python -m pip install -e /workspace/third_parties/RFdiffusion
RUN rm -rf /workspace/third_parties/RFdiffusion

ENTRYPOINT []

WORKDIR /workspace
