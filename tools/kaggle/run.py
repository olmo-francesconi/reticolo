#!/usr/bin/env python3
"""Kaggle GPU kernel body — build & test the reticolo CUDA backend.

Runs headless on a Kaggle GPU kernel (needs enable_gpu + enable_internet in the
kernel metadata; push.sh sets both). Clones the CUDA branch and runs the shared
tools/cuda_build_test.sh, so the build/test logic lives in exactly one place.
"""
import subprocess

BRANCH = "feat/cuda-extension"
REPO = "https://github.com/olmo-francesconi/reticolo.git"


def sh(cmd):
    print(f"+ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh("nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv")
sh("nvcc --version")
sh("pip install -q 'cmake>=3.25' ninja")
sh(f"rm -rf reticolo && git clone --depth 1 --branch {BRANCH} {REPO}")
sh("cd reticolo && bash tools/cuda_build_test.sh")
print("CUDA build & test OK", flush=True)
