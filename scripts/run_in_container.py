#!/usr/bin/env python3
"""Run a command inside a throwaway Docker container and clean up afterwards."""

from __future__ import annotations

import argparse
import subprocess
import uuid
import shlex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Container image to run")
    parser.add_argument("--tag", default="latest", help="Image tag (default: latest)")
    parser.add_argument("--name-prefix", default="profile-sweep", help="Prefix for temporary container name")
    parser.add_argument("--gpus", default=None, help="Value to pass to --gpus (omit to disable)")
    parser.add_argument("--workdir", default="/workspace", help="Working directory inside the container")
    parser.add_argument("--mount", action="append", default=[], help="Mount specification host:container[:options]")
    parser.add_argument("--env", action="append", default=[], help="Environment variable to set inside container (KEY=VALUE)")
    parser.add_argument("--extra-arg", action="append", default=[], help="Additional raw docker run argument (single token)")
    parser.add_argument("--quiet", action="store_true", help="Suppress printing the docker command")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to execute inside the container")
    args = parser.parse_args()
    if not args.command:
        parser.error("No command specified for container execution (provide arguments after --)")
    return args


def build_docker_command(args: argparse.Namespace) -> list[str]:
    container_name = f"{args.name_prefix}-{uuid.uuid4().hex[:8]}"
    docker_cmd: list[str] = ["docker", "run", "--rm", "--name", container_name]

    for extra in args.extra_arg:
        docker_cmd.append(extra)

    if args.gpus:
        docker_cmd.extend(["--gpus", args.gpus])

    if args.workdir:
        docker_cmd.extend(["-w", args.workdir])

    for mount in args.mount:
        docker_cmd.extend(["-v", mount])

    for env_var in args.env:
        docker_cmd.extend(["-e", env_var])

    command_parts = args.command
    if command_parts and command_parts[0] == '--':
        command_parts = command_parts[1:]
    command_str = " ".join(shlex.quote(part) for part in command_parts)

    docker_cmd.extend([
        f"{args.image}:{args.tag}",
        "bash",
        "-lc",
        command_str,
    ])

    return docker_cmd


def main() -> int:
    args = parse_args()
    docker_cmd = build_docker_command(args)

    if not args.quiet:
        print("[INFO] docker command:", " ".join(shlex.quote(part) for part in docker_cmd))

    completed = subprocess.run(docker_cmd)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
