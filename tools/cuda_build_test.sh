#!/usr/bin/env bash
# Configure, build, and test the reticolo CUDA backend on a GPU host.
#
# Assumes it is run from the repo root with nvcc, a C++ toolchain, CMake >= 3.25
# and Ninja on PATH. The CUDA backend includes HDF5 output +
# checkpoint and native CUDA apps, so IO + APPS are ON by default and HDF5 +
# cxxopts are required (the runner installs libhdf5-dev; cxxopts is
# FetchContent'd). Set RETICOLO_BUILD_IO=OFF for a lean core+cuda smoke that
# needs neither (APPS follows IO unless set explicitly).
#
# Usage:   tools/cuda_build_test.sh [build_dir]
# Env:     RETICOLO_CUDA_ARCH    CUDA arch (default: native — the host GPU)
#          RETICOLO_TEST_FILTER  ctest -R regex (default: cuda)
#          RETICOLO_BUILD_IO     ON/OFF (default: ON)
#          RETICOLO_BUILD_APPS   ON/OFF (default: follows IO)
#
# OpenMP is OFF so reticolo::core pulls no -fopenmp interface flags onto the
# nvcc-compiled apps, and ReticoloWarnings guards -Wall/-Werror behind
# $<COMPILE_LANGUAGE:CXX> so they never reach nvcc either — that is what lets a
# .cu app link reticolo::io/cli (the umbrella) cleanly. Only the CUDA-relevant
# targets are built (not the whole suite); ctest -R cuda runs exactly the
# discovered CUDA cases (built targets register via catch_discover_tests).
#
# This is the headless core shared by the Colab notebook and any console runner
# (Kaggle CLI, a GPU GitHub Actions runner, papermill, an SSH'd GPU VM).
set -euo pipefail

build_dir="${1:-build/cuda}"
arch="${RETICOLO_CUDA_ARCH:-native}"
filter="${RETICOLO_TEST_FILTER:-cuda}"
io="${RETICOLO_BUILD_IO:-ON}"
apps="${RETICOLO_BUILD_APPS:-${io}}"

export PATH="/usr/local/cuda/bin:${PATH}"

cmake -S . -B "${build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DRETICOLO_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${arch}" \
  -DRETICOLO_BUILD_TESTS=ON \
  -DRETICOLO_BUILD_IO="${io}" \
  -DRETICOLO_BUILD_APPS="${apps}" \
  -DRETICOLO_BUILD_EXAMPLES=OFF \
  -DRETICOLO_ENABLE_OPENMP=OFF \
  -DRETICOLO_WARNINGS_AS_ERRORS=ON

# Build only the CUDA-relevant targets. The self-test always; with IO the
# checkpoint round-trip; with IO+APPS the native CUDA apps + their HDF5 smoke
# (the smoke target pulls phi4_cuda_hmc as a build dependency).
targets=(test_cuda_selftest)
if [[ "${io}" == "ON" ]]; then
  targets+=(test_cuda_checkpoint)
fi
if [[ "${io}" == "ON" && "${apps}" == "ON" ]]; then
  targets+=(phi4_cuda_hmc su2_cuda_hmc test_phi4_cuda_hmc_smoke)
fi

cmake --build "${build_dir}" --target "${targets[@]}" -j "$(nproc 2>/dev/null || echo 2)"
ctest --test-dir "${build_dir}" -R "${filter}" --output-on-failure

# Optional GPU perf baseline (RETICOLO_BENCH=1): build + run the bench binary.
if [[ -n "${RETICOLO_BENCH:-}" ]]; then
  echo "=== bench_cuda_hmc ==="
  cmake --build "${build_dir}" --target bench_cuda_hmc -j "$(nproc 2>/dev/null || echo 2)"
  "${build_dir}/src/cuda/bench_cuda_hmc"
fi

# Optional nightly physics harness (RETICOLO_NIGHTLY=1, needs IO+APPS): build +
# run the CPU-vs-GPU identity checks. Its non-zero exit fails the run.
if [[ -n "${RETICOLO_NIGHTLY:-}" ]]; then
  echo "=== cuda_nightly ==="
  cmake --build "${build_dir}" --target cuda_nightly -j "$(nproc 2>/dev/null || echo 2)"
  "${build_dir}/apps/cuda_nightly"
fi
