#!/usr/bin/env python3
"""Kaggle GPU kernel body — build & test the reticolo CUDA backend.

Runs headless on a Kaggle GPU kernel (needs enable_gpu + enable_internet in the
kernel metadata; push.sh sets both). Clones the CUDA branch and runs the shared
tools/cuda_build_test.sh, so the build/test logic lives in exactly one place.

Kaggle's default host compiler is GCC 11, which predates C++23 <format> (used by
reticolo's core logger). We install GCC 13 and point both the C++ and the nvcc
host compiler at it via the CC/CXX/CUDAHOSTCXX env vars CMake honours.
"""
import os
import subprocess

BRANCH = "feat/cuda-extension"
REPO = "https://github.com/olmo-francesconi/reticolo.git"


def sh(cmd):
    print(f"+ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh("nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv")
sh("nvcc --version")

# GCC 13 for C++23 <format> (Kaggle ships GCC 11, which lacks the header).
os.environ["DEBIAN_FRONTEND"] = "noninteractive"
sh("apt-get -qq update")
sh("apt-get -qq install -y software-properties-common")
sh("add-apt-repository -y ppa:ubuntu-toolchain-r/test")
sh("apt-get -qq update")
sh("apt-get -qq install -y gcc-13 g++-13")
# libhdf5-dev for reticolo::io (Phase 6 CUDA backend writes HDF5 + checkpoints).
sh("apt-get -qq install -y libhdf5-dev")

sh("pip install -q 'cmake>=3.25' ninja")
# Clone and build under /tmp, NOT /kaggle/working — the latter is persisted as
# the kernel's "output", so building there makes `kaggle kernels output` try to
# download the whole Sleef/Catch2/object tree (100s of MB). Building in /tmp
# keeps the output dir empty, so the log download is instant.
sh(f"rm -rf /tmp/reticolo && git clone --depth 1 --branch {BRANCH} {REPO} /tmp/reticolo")

os.environ["CC"] = "gcc-13"
os.environ["CXX"] = "g++-13"
os.environ["CUDAHOSTCXX"] = "g++-13"
os.environ["RETICOLO_BENCH"] = "1"    # also build + run the GPU perf baseline
os.environ["RETICOLO_NIGHTLY"] = "1"  # and the CPU-vs-GPU physics harness
sh("cd /tmp/reticolo && bash tools/cuda_build_test.sh")
print("CUDA build & test OK", flush=True)
