#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-alphafold3}"
TAG="${TAG:-latest}"

MODEL_SRC="$PROJECT_ROOT/third_parties/alphafold3"
DOCKERFILE_PATH="$MODEL_SRC/docker/Dockerfile"

if [[ ! -f "$DOCKERFILE_PATH" ]]; then
  echo "[ERROR] Dockerfile not found: $DOCKERFILE_PATH" >&2
  exit 1
fi

echo "[INFO] Building Docker image: ${IMAGE_NAME}:${TAG}"
echo "[INFO] Using Dockerfile: $DOCKERFILE_PATH"
echo "[INFO] Build context: $MODEL_SRC"

docker build \
  -t "${IMAGE_NAME}:${TAG}" \
  -f "$DOCKERFILE_PATH" \
  "$MODEL_SRC"