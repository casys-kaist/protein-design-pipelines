#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-autodock_vina_gpu}"
TAG="${TAG:-latest}"
SOURCE_IMAGE="${SOURCE_IMAGE:-fovus/vina-gpu-2.1:autodock-vina-gpu}"

docker pull "${SOURCE_IMAGE}"
docker tag "${SOURCE_IMAGE}" "${IMAGE_NAME}:${TAG}"
