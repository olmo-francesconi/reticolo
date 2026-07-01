#!/usr/bin/env bash
# One-time environment bootstrap on a fresh RunPod pod. Idempotent — safe to
# re-run. Installs everything the reticolo CUDA build + nsys profiling needs on
# top of a CUDA-devel / PyTorch-devel image. See tools/runpod/README.md.
#
#   ./tools/runpod/setup.sh
set -euo pipefail
. "$(dirname "$0")/common.sh"

echo ">> pod: $RUNPOD_HOST:$RUNPOD_PORT  (arch sm_$RUNPOD_ARCH)"
pod bash -s <<'REMOTE'
set -e
export DEBIAN_FRONTEND=noninteractive

echo "== GPU =="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

echo "== CUDA toolkit =="
CUDA_BIN="$(for d in $(ls -d /usr/local/cuda-*/bin 2>/dev/null | sort -Vr) /usr/local/cuda/bin; do [ -x "$d/nvcc" ] && { echo "$d"; break; }; done)"
[ -n "$CUDA_BIN" ] || { echo "FATAL: no nvcc found — use a CUDA-devel image, not runtime-only"; exit 1; }
export PATH="$CUDA_BIN:/usr/local/bin:$PATH"
nvcc --version | grep release

echo "== apt deps (ninja, rsync, hdf5) =="
apt-get update -qq
apt-get install -y -qq ninja-build rsync libhdf5-dev

echo "== cmake >= 3.25 (pip) =="
if ! cmake --version 2>/dev/null | head -1 | grep -qE "3\.(2[5-9]|[3-9][0-9])|[4-9]\."; then
  python3 -m pip install -q --upgrade "cmake>=3.25"
fi
hash -r; cmake --version | head -1

echo "== g++-13 (Ubuntu 22.04 default GCC 11 lacks <format>) =="
if ! g++-13 --version >/dev/null 2>&1; then
  # add-apt-repository is broken when pip shadows the system python's apt_pkg,
  # so add the toolchain PPA by hand (jammy).
  echo "deb https://ppa.launchpadcontent.net/ubuntu-toolchain-r/test/ubuntu jammy main" > /etc/apt/sources.list.d/toolchain.list
  gpg --keyserver keyserver.ubuntu.com --recv-keys 1E9377A2BA9EF27F 2>/dev/null
  gpg --export 1E9377A2BA9EF27F > /etc/apt/trusted.gpg.d/toolchain.gpg
  apt-get update -qq
  apt-get install -y -qq g++-13
fi
g++-13 --version | head -1

echo "== nsys =="
if ! command -v nsys >/dev/null 2>&1; then
  # Version from nvcc ("release 12.4" -> 12-4), NOT the cuda-*/bin dir name: that
  # can be a major-only symlink (cuda-12) and yield the wrong package. Refresh the
  # CUDA repo index first — setup's earlier apt-get update may predate it, leaving
  # cuda-nsight-systems-* "Unable to locate".
  VER="$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1-\2/p')"
  apt-get update -qq
  apt-get install -y -qq "cuda-nsight-systems-$VER" || apt-get install -y -qq nsight-systems-cli || true
fi
if ! command -v nsys >/dev/null 2>&1; then
  # Fallback: reuse the nsys bundled inside nsight-compute (ships in most devel
  # images) by symlinking it onto PATH.
  BUNDLED="$(find /opt/nvidia/nsight-compute -name nsys -type f 2>/dev/null | head -1)"
  [ -n "$BUNDLED" ] && ln -sf "$BUNDLED" /usr/local/bin/nsys
fi
hash -r
nsys --version 2>/dev/null | head -1 || echo "nsys NOT installed"

echo "== ncu (blocked in pod containers by ERR_NVGPUCTRPERM — nsys is the usable one) =="
[ -x "$CUDA_BIN/ncu" ] && echo "ncu present (counters likely blocked)" || echo "ncu absent"

echo "== done =="
REMOTE
echo ">> setup complete. Next: ./tools/runpod/build.sh"
