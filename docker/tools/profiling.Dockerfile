ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN set -eux \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && . /etc/os-release \
    && arch="$(dpkg --print-architecture)" \
    && keyring="/usr/share/keyrings/nvidia-profiler.gpg" \
    && mkdir -p "$(dirname "$keyring")" \
    && if [ "$ID" = "ubuntu" ]; then \
        version="$(printf '%s' "$VERSION_ID" | tr -d '.')" ; \
        repo="https://developer.download.nvidia.com/devtools/repos/ubuntu${version}/${arch}" ; \
        curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub" | gpg --dearmor -o "$keyring" ; \
        echo "deb [signed-by=$keyring] $repo /" > /etc/apt/sources.list.d/nvidia-profiler.list ; \
        apt-get update ; \
        apt-get install -y --no-install-recommends nsight-systems-cli nsight-compute ; \
    elif [ "$ID" = "debian" ]; then \
        version="$(printf '%s' "$VERSION_ID" | tr -d '.')" ; \
        repo_arch="x86_64" ; \
        if [ "$arch" != "amd64" ]; then repo_arch="$arch" ; fi ; \
        repo="https://developer.download.nvidia.com/compute/cuda/repos/debian${version}/${repo_arch}" ; \
        curl -fsSL "${repo}/3bf863cc.pub" | gpg --dearmor -o "$keyring" ; \
        echo "deb [signed-by=$keyring] $repo /" > /etc/apt/sources.list.d/nvidia-profiler.list ; \
        apt-get update ; \
        nsight_systems_pkg="$(apt-cache search --names-only '^nsight-systems-[0-9]' | awk '{print $1}' | sort -V | tail -n1)" ; \
        nsight_compute_pkg="$(apt-cache search --names-only '^nsight-compute-[0-9]' | awk '{print $1}' | sort -V | tail -n1)" ; \
        if [ -z "$nsight_systems_pkg" ] || [ -z "$nsight_compute_pkg" ]; then \
            echo "Unable to resolve Nsight packages for Debian ${VERSION_ID}" >&2 ; \
            exit 1 ; \
        fi ; \
        apt-get install -y --no-install-recommends "$nsight_systems_pkg" "$nsight_compute_pkg" ; \
    else \
        echo "Unsupported distribution: ${ID}" >&2 ; \
        exit 1 ; \
    fi \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

    # Create a stable symlink for Nsight Compute (version-agnostic)
RUN set -eux \
    && root="/opt/nvidia/nsight-compute" \
    && dir="$(ls -d ${root}/* 2>/dev/null | sort -V | tail -n1)" \
    && test -n "$dir" && test -x "$dir/ncu" \
    && ln -s "$dir" "${root}/current" \
    && ln -sf "${root}/current/ncu" /usr/local/bin/ncu

RUN set -eux \
    && if command -v micromamba >/dev/null 2>&1; then \
        env_name="${CONDA_DEFAULT_ENV:-base}" ; \
        micromamba run -n "${env_name}" python -m pip install --no-cache-dir pynvml tqdm hydra-core psutil pyyaml ; \
    elif command -v python >/dev/null 2>&1; then \
        python -m pip install --no-cache-dir pynvml tqdm hydra-core psutil pyyaml ; \
    elif command -v python3 >/dev/null 2>&1; then \
        python3 -m pip install --no-cache-dir pynvml tqdm hydra-core psutil pyyaml ; \
    else \
        echo "Unable to locate a Python interpreter for pip install." >&2 ; \
        exit 1 ; \
    fi \
    && rm -rf /root/.cache/pip
