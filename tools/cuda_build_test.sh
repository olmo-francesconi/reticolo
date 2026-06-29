#!/usr/bin/env bash
# Configure, build, and test the reticolo CUDA backend on a GPU host.
#
# Assumes it is run from the repo root with nvcc, a C++ toolchain, CMake >= 3.25
# and Ninja on PATH. Phase 0 needs only reticolo::core + reticolo::cuda, so IO /
# apps / examples are off and HDF5 is not required.
#
# Usage:   tools/cuda_build_test.sh [build_dir]
# Env:     RETICOLO_CUDA_ARCH   CUDA arch (default: native — the host GPU)
#          RETICOLO_TEST_FILTER ctest -R regex (default: cuda)
#
# This is the headless core shared by the Colab notebook and any console runner
# (Kaggle CLI, a GPU GitHub Actions runner, papermill, an SSH'd GPU VM).
set -euo pipefail

build_dir="${1:-build/cuda}"
arch="${RETICOLO_CUDA_ARCH:-native}"
filter="${RETICOLO_TEST_FILTER:-cuda}"

export PATH="/usr/local/cuda/bin:${PATH}"

cmake -S . -B "${build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DRETICOLO_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${arch}" \
  -DRETICOLO_BUILD_TESTS=ON \
  -DRETICOLO_BUILD_IO=OFF \
  -DRETICOLO_BUILD_APPS=OFF \
  -DRETICOLO_BUILD_EXAMPLES=OFF \
  -DRETICOLO_ENABLE_OPENMP=OFF \
  -DRETICOLO_WARNINGS_AS_ERRORS=ON

cmake --build "${build_dir}" -j "$(nproc 2>/dev/null || echo 2)"
ctest --test-dir "${build_dir}" -R "${filter}" --output-on-failure
</content>
