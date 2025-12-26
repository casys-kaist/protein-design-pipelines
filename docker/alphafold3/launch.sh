#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
IMAGE=alphafold3
TAG=latest
NAME="$(whoami)_alphafold3"
MNT_DIR=/mnt

docker run -it \
    --net=host \
    -v ${SCRIPT_DIR}/../..:/workspace \
    -v ${MNT_DIR}/nfs:/mnt/nfs \
    -v ${MNT_DIR}/ssd1:/mnt/ssd1 \
    -v ${MNT_DIR}/ssd2:/mnt/ssd2 \
    -v ${MNT_DIR}/ssd3:/mnt/ssd3 \
    -v ${MNT_DIR}/ssd4:/mnt/ssd4 \
    -v ${MNT_DIR}/hdd1:/mnt/hdd1 \
    -v ${MNT_DIR}/hdd2:/mnt/hdd2 \
    -v ${MNT_DIR}/ssd:/mnt/ssd \
    --gpus all \
    --ulimit core=-1 --privileged \
    --ulimit memlock=-1 \
    --security-opt seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    --cap-add=IPC_LOCK \
    --shm-size=110gb \
    --name=${NAME} \
    ${IMAGE}:${TAG} /bin/bash
