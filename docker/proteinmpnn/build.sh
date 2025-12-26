#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-proteinmpnn}"
TAG="${TAG:-latest}"

docker build -t "${IMAGE_NAME}:${TAG}" "$SCRIPT_DIR"
