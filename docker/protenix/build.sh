#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-protenix}"
TAG="${TAG:-latest}"

REQ_SRC="$PROJECT_ROOT/third_parties/Protenix/requirements.txt"
REQ_DST="$SCRIPT_DIR/requirements.txt"

if [[ ! -f "$REQ_SRC" ]]; then
  echo "[ERROR] Expected requirements file not found: $REQ_SRC" >&2
  exit 1
fi

cleanup() {
  rm -f "$REQ_DST"
}
trap cleanup EXIT

cp "$REQ_SRC" "$REQ_DST"

docker build -t "${IMAGE_NAME}:${TAG}" "$SCRIPT_DIR"
