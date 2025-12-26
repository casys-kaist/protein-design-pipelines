#!/bin/bash
set -euo pipefail

NAME="$(whoami)_alphafold3"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker command not found. Install Docker first." >&2
    exit 1
fi

if ! docker inspect "$NAME" >/dev/null 2>&1; then
    echo "Container '$NAME' not found. Run ${SCRIPT_DIR}/launch.sh first." >&2
    exit 1
fi

state=$(docker inspect -f '{{.State.Running}}' "$NAME")
if [ "$state" != "true" ]; then
    echo "Container '$NAME' is not running. Start it (e.g. rerun ${SCRIPT_DIR}/launch.sh) and try again." >&2
    exit 1
fi

if [[ ${CHECK_NSIGHT:-0} == 1 ]]; then
    docker exec "$NAME" bash -lc '
missing=0
for tool in nsys ncu; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "[warn] $tool not found" >&2
    missing=1
  else
    echo "[ok] $tool available"
  fi
done
if [ "$missing" -ne 0 ]; then
  echo "[warn] Profiling tools missing. Run /workspace/docker/tools/install_nsight_tools.sh inside the container to install." >&2
fi
'
fi

if [ "$#" -eq 0 ]; then
    docker exec -it "$NAME" bash
else
    docker exec -it "$NAME" "$@"
fi
