#!/bin/bash
# Install Nsight Systems (nsys) and Nsight Compute (ncu) CLI tools inside a
# container image.

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    echo "[ERROR] Run this script as root inside the container (try: sudo bash $0)." >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive

apt update
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt update