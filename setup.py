#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from setuptools import find_packages, setup
from pathlib import Path

requirements_file = Path(__file__).parent / "requirements.txt"
setup(
    name="slowfast",
    version="1.0",
    author="FAIR",
    url="unknown",
    description="SlowFast Video Understanding",
    install_requires=[
        requirements_file.read_text().splitlines()
    ],
    extras_require={"tensorboard_video_visualization": ["moviepy"]},
    packages=find_packages(exclude=("configs", "tests")),
)
