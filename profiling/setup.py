#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="profile-sweeps",
    version="0.1.0",
    description="Hydra-based runners for profiling component sweeps",
    author="",
    author_email="",
    url="https://github.com/user/project",
    install_requires=[
        "hydra-core>=1.3",
        "omegaconf>=2.3",
        "psutil>=5.9",
        "pynvml",
        "tqdm>=4.65",
        "pyyaml>=6.0",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "profile-sweeps = profile.cli.run_sweeps:main",
        ]
    },
)
