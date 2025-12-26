#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-esm-2}"
TAG="${TAG:-latest}"
SOURCE_IMAGE="${SOURCE_IMAGE:-protenix:latest}"

if ! docker image inspect "${SOURCE_IMAGE}" >/dev/null 2>&1; then
  echo "[INFO] Source image '${SOURCE_IMAGE}' not present locally; attempting to pull." >&2
  if ! docker pull "${SOURCE_IMAGE}"; then
    echo "[ERROR] Failed to locate source image '${SOURCE_IMAGE}'. Build the protenix image first (e.g. python docker/build.py protenix) or set SOURCE_IMAGE." >&2
    exit 1
  fi
fi

docker tag "${SOURCE_IMAGE}" "${IMAGE_NAME}:${TAG}"
