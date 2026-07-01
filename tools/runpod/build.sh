#!/usr/bin/env bash
# Sync + configure + build + ctest on the pod — the gate, via the linux-nvcc
# preset with the pod's arch and g++-13 host compiler.
#
#   ./tools/runpod/build.sh              # build all + run ctest
#   ./tools/runpod/build.sh <target>     # build just one target, skip ctest
set -euo pipefail
_dir="$(dirname "$0")"
. "$_dir/common.sh"
"$_dir/sync.sh"

TARGET="${1:-}"
echo ">> configure + build (sm_$RUNPOD_ARCH, g++-13)${TARGET:+, target=$TARGET}"
pod_run DIR="$RUNPOD_DIR" ARGS="$CMAKE_ARGS" TARGET="$TARGET" <<'REMOTE'
set -e
export PATH="$(for d in $(ls -d /usr/local/cuda-*/bin 2>/dev/null | sort -Vr) /usr/local/cuda/bin; do [ -x "$d/nvcc" ] && { echo "$d"; break; }; done):/usr/local/bin:$PATH"
cd "$DIR"
cmake --preset linux-nvcc $ARGS
if [ -n "$TARGET" ]; then
  cmake --build --preset linux-nvcc --target "$TARGET" -j "$(nproc)"
else
  cmake --build --preset linux-nvcc -j "$(nproc)"
  ctest --preset linux-nvcc --output-on-failure
fi
REMOTE
echo ">> build done"
