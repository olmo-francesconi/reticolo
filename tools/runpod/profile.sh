#!/usr/bin/env bash
# Sync + build one target + profile it under nsys, then pull the .nsys-rep (opens
# in the Nsight Systems GUI) + kernel/api/memory CSVs to tools/runpod/output/prof/.
#
# --cuda-graph-trace=node is essential: cuda::Hmc replays each trajectory as a
# captured CUDA graph, so without it nsys shows one opaque "graph launch" per
# trajectory instead of the per-kernel breakdown. Kernel names carry the semantics.
#
# NOTE: ncu (occupancy/roofline) is blocked in pod containers (ERR_NVGPUCTRPERM);
# nsys is the usable profiler. Keep the run SHORT — graph-node tracing makes the
# trace grow fast (a few hundred trajectories is plenty for the breakdown).
#
#   ./tools/runpod/profile.sh phi4_cuda_hmc --size=16 --n_therm=20 --n_prod=200 --out=p.h5
set -euo pipefail
_dir="$(dirname "$0")"
. "$_dir/common.sh"
[ $# -ge 1 ] || { echo "usage: profile.sh <target> [app flags...]"; exit 1; }
TARGET="$1"; shift; APP_ARGS="$*"

"$_dir/build.sh" "$TARGET"

echo ">> nsys profile $TARGET $APP_ARGS"
pod_run DIR="$RUNPOD_DIR" TARGET="$TARGET" APP_ARGS="$APP_ARGS" <<'REMOTE'
set -e
export PATH="$(for d in $(ls -d /usr/local/cuda-*/bin 2>/dev/null | sort -Vr) /usr/local/cuda/bin; do [ -x "$d/nvcc" ] && { echo "$d"; break; }; done):/usr/local/bin:$PATH"
cd "$DIR"
BIN="$PWD/$(find build/linux-nvcc -name "$TARGET" -type f -perm -u+x | head -1)"
[ -x "$BIN" ] || { echo "no binary for $TARGET"; exit 1; }
rm -rf /tmp/reticolo_prof && mkdir -p /tmp/reticolo_prof && cd /tmp/reticolo_prof
nsys profile --trace=cuda --cuda-graph-trace=node --force-overwrite=true -o prof "$BIN" $APP_ARGS
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum --format csv --output stats prof.nsys-rep >/dev/null
echo "=== top kernels (% GPU time) ==="
head -12 stats_cuda_gpu_kern_sum.csv | awk -F, '{printf "%-6s %-9s %s\n", $1, $3, $NF}'
ls -la prof.nsys-rep
REMOTE

OUT="$REPO/tools/runpod/output/prof"; mkdir -p "$OUT"
echo ">> pulling report + CSVs -> $OUT"
pod_scp "root@$RUNPOD_HOST:/tmp/reticolo_prof/prof.nsys-rep" "$OUT/" 2>/dev/null || true
pod_scp "root@$RUNPOD_HOST:/tmp/reticolo_prof/stats_*.csv" "$OUT/" 2>/dev/null || true
echo ">> open in Nsight Systems:  $OUT/prof.nsys-rep"
ls -la "$OUT"
