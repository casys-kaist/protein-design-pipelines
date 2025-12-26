#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-diffdock}"
TAG="${TAG:-latest}"

ENV_SRC="$PROJECT_ROOT/third_parties/DiffDock/environment.yml"
ENV_DST="$SCRIPT_DIR/environment.yml"

if [[ ! -f "$ENV_SRC" ]]; then
  echo "[ERROR] Expected environment file not found: $ENV_SRC" >&2
  exit 1
fi

cleanup() {
  rm -f "$ENV_DST"
}
trap cleanup EXIT

cp "$ENV_SRC" "$ENV_DST"

docker build -t "${IMAGE_NAME}:${TAG}" "$SCRIPT_DIR"
