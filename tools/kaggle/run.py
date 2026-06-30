#!/usr/bin/env python3
"""Kaggle GPU kernel body — build & test the reticolo CUDA backend.

Runs headless on a Kaggle GPU kernel (needs enable_gpu + enable_internet in the
kernel metadata; push.sh sets both). The repo is shipped as a Kaggle *dataset*
(a tarball of the local working tree, uploaded by push.sh) and extracted here, so
a run reflects the local checkout — including uncommitted changes — with no git
commit/push. The build/test logic lives in tools/cuda_build_test.sh +
tools/cuda_profile.sh (carried in the tarball), so it stays in one place.

Kaggle's default host compiler is GCC 11, which predates C++23 <format> (used by
reticolo's core logger). We install GCC 13 and point both the C++ and the nvcc
host compiler at it via the CC/CXX/CUDAHOSTCXX env vars CMake honours.
"""
import glob
import os
import subprocess

# Per-run toggles (push.sh ships THIS local file, so editing these needs no
# commit). Gates = 47 ctest cases + reversibility + bench + CPU-vs-GPU nightly;
# Profile = Nsight sweep + lb_sweep + f32 throughput.
RUN_GATES = True
RUN_PROFILE = False

INP = "/kaggle/input/reticolo-src"


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
# libhdf5-dev for reticolo::io (the CUDA backend writes HDF5 + checkpoints).
sh("apt-get -qq install -y libhdf5-dev")

sh("pip install -q 'cmake>=3.25' ninja")
# Extract the local snapshot under /tmp, NOT /kaggle/working — the latter is
# persisted as the kernel's "output", so building there makes `kaggle kernels
# output` try to download the whole Sleef/Catch2/object tree (100s of MB).
# Building in /tmp keeps the output dir empty, so the log download is instant.
sh(f"ls -la {INP}")
sh("rm -rf /tmp/reticolo && mkdir -p /tmp/reticolo")
# Kaggle may auto-extract the uploaded tarball: if the archive is kept, extract
# it; if it was already unpacked into the dataset dir, copy the tree as-is.
tgz = glob.glob(f"{INP}/*.tar.gz")
if tgz:
    sh(f"tar xzf {tgz[0]} -C /tmp/reticolo")
else:
    sh(f"cp -a {INP}/. /tmp/reticolo/")

os.environ["CC"] = "gcc-13"
os.environ["CXX"] = "g++-13"
os.environ["CUDAHOSTCXX"] = "g++-13"

if RUN_GATES:
    os.environ["RETICOLO_BENCH"] = "1"    # also build + run the GPU perf baseline
    os.environ["RETICOLO_NIGHTLY"] = "1"  # and the CPU-vs-GPU physics harness
    sh("cd /tmp/reticolo && bash tools/cuda_build_test.sh")
    print("CUDA build & test OK", flush=True)

if RUN_PROFILE:
    # Nsight profiling pass (phi4 + su3 scaling). Writes nsys/ncu reports + CSVs +
    # throughput.jsonl + lb_sweep.jsonl to /kaggle/working/profile for download.
    sh("cd /tmp/reticolo && bash tools/cuda_profile.sh /tmp/reticolo/build/cuda "
       "/kaggle/working/profile")
    print("CUDA profiling OK", flush=True)
