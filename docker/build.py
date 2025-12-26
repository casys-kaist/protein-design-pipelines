#!/usr/bin/env python3
"""Build component docker images and add a profiling layer."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


SCRIPT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = SCRIPT_DIR / "tools"
PROFILING_DOCKERFILE = TOOLS_DIR / "profiling.Dockerfile"
BASE_TAG = "latest"
PROFILE_TAG = "profiling"
ALL_KEYWORD = "all"


def list_images() -> List[str]:
    names: List[str] = []
    for child in SCRIPT_DIR.iterdir():
        if not child.is_dir():
            continue
        if (child / "build.sh").is_file():
            names.append(child.name)
    return sorted(names)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="docker/build.py",
        description="Build component images and layer profiling tools.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available component images.",
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Component name (directory under docker/) to build, or 'all' to build everything from --list.",
    )
    return parser.parse_args(argv)


def docker_image_exists(ref: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", ref],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def build_component(image: str, tag: str) -> None:
    component_dir = SCRIPT_DIR / image
    script_path = component_dir / "build.sh"
    if not script_path.exists():
        raise RuntimeError(f"Component build script missing for '{image}'.")

    print(f"[INFO] Building {image}:{tag} via {script_path}", file=sys.stderr)
    env = os.environ.copy()
    env["TAG"] = tag
    subprocess.run([str(script_path)], check=True, env=env)


def build_profiling_layer(image: str, profile_tag: str) -> None:
    if profile_tag == BASE_TAG:
        raise RuntimeError("Profiling tag must differ from the base tag.")
    if not PROFILING_DOCKERFILE.exists():
        raise RuntimeError(f"Profiling Dockerfile missing at {PROFILING_DOCKERFILE}")

    base_ref = f"{image}:{BASE_TAG}"
    if not docker_image_exists(base_ref):
        print(f"[INFO] Base image {base_ref} missing; building it first.", file=sys.stderr)
        build_component(image, tag=BASE_TAG)

    profile_ref = f"{image}:{profile_tag}"
    cmd: List[str] = [
        "docker",
        "build",
        "-f",
        str(PROFILING_DOCKERFILE),
        "-t",
        profile_ref,
        "--build-arg",
        f"BASE_IMAGE={base_ref}",
        str(TOOLS_DIR),
    ]

    print(f"[INFO] Building profiling image {profile_ref} from {base_ref}", file=sys.stderr)
    env = os.environ.copy()
    env.setdefault("DOCKER_BUILDKIT", "1")
    subprocess.run(cmd, check=True, env=env)


def reorder_images_for_dependencies(images: List[str]) -> List[str]:
    """Ensure dependent images build in a sensible order."""

    if "protenix" in images and "esm-2" in images:
        protenix_idx = images.index("protenix")
        esm2_idx = images.index("esm-2")
        if protenix_idx > esm2_idx:
            reordered = images.copy()
            reordered.pop(protenix_idx)
            reordered.insert(esm2_idx, "protenix")
            return reordered

    return images


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    if args.list:
        for name in list_images():
            print(name)
        return 0

    if not args.image:
        print("[ERROR] No image specified.", file=sys.stderr)
        return 1

    if args.image == ALL_KEYWORD:
        images = reorder_images_for_dependencies(list_images())
        if not images:
            print("[ERROR] No images found under docker/.", file=sys.stderr)
            return 1
    else:
        component_dir = SCRIPT_DIR / args.image
        if not component_dir.exists():
            print(f"[ERROR] Unknown image '{args.image}'.", file=sys.stderr)
            return 1
        images = [args.image]

    failures: List[str] = []
    for image in images:
        try:
            build_profiling_layer(image, PROFILE_TAG)
        except RuntimeError as exc:
            print(f"[ERROR] {image}: {exc}", file=sys.stderr)
            failures.append(image)
        except subprocess.CalledProcessError as exc:
            cmd = getattr(exc, "cmd", "<external command>")
            if isinstance(cmd, (list, tuple)):
                cmd = " ".join(str(part) for part in cmd)
            print(
                f"[ERROR] {image}: command failed ({cmd}) with exit code {exc.returncode}",
                file=sys.stderr,
            )
            failures.append(image)

    if failures:
        print(f"[ERROR] Failed images: {', '.join(failures)}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
