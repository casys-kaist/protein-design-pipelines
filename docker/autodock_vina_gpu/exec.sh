#!/bin/bash
set -euo pipefail

usage() {
    cat <<USAGE
Usage: ${0##*/} [--tag TAG] [--] [COMMAND [ARGS...]]

Run a command inside the autodock-vina-gpu container. If no COMMAND is
provided, an interactive bash shell is opened.
USAGE
}

TAG="autodock-vina-gpu"
declare -a CMD=()

while (($#)); do
    case "$1" in
        --tag)
            if (($# < 2)); then
                echo "[ERROR] --tag requires a value" >&2
                exit 1
            fi
            TAG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            CMD=("$@")
            break
            ;;
        *)
            CMD=("$@")
            break
            ;;
    esac
done

if [ "$TAG" = "autodock-vina-gpu" ]; then
    SUFFIX=""
else
    SUFFIX="_${TAG}"
fi

NAME="$(whoami)_vinagpu${SUFFIX}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] docker command not found. Install Docker first." >&2
    exit 1
fi

if ! docker inspect "$NAME" >/dev/null 2>&1; then
    if [ "$TAG" = "autodock-vina-gpu" ]; then
        hint="${SCRIPT_DIR}/launch.sh"
    else
        hint="${SCRIPT_DIR}/launch.sh ${TAG}"
    fi
    echo "[ERROR] Container '$NAME' not found. Launch it via: ${hint}" >&2
    exit 1
fi

state=$(docker inspect -f '{{.State.Running}}' "$NAME")
if [ "$state" != "true" ]; then
    echo "[ERROR] Container '$NAME' is not running. Start it (e.g. rerun launch.sh) and retry." >&2
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

if [ ${#CMD[@]} -eq 0 ]; then
    docker exec -it "$NAME" bash
else
    docker exec -it "$NAME" "${CMD[@]}"
fi
