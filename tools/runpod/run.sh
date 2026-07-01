#!/usr/bin/env bash
# Sync + build one target + run it on the pod, then pull its output files back
# to tools/runpod/output/<target>/. Pass app flags after the target; use a bare
# --out filename (no --workspace) so the file lands in the run dir we pull.
#
#   ./tools/runpod/run.sh phi4_cuda_hmc --size=8 --n_prod=200 --out=x.h5
set -euo pipefail
_dir="$(dirname "$0")"
. "$_dir/common.sh"
[ $# -ge 1 ] || { echo "usage: run.sh <target> [app flags...]"; exit 1; }
TARGET="$1"; shift; APP_ARGS="$*"

"$_dir/build.sh" "$TARGET"

echo ">> run $TARGET $APP_ARGS"
pod_run DIR="$RUNPOD_DIR" TARGET="$TARGET" APP_ARGS="$APP_ARGS" <<'REMOTE'
set -e
cd "$DIR"
BIN="$PWD/$(find build/linux-nvcc -name "$TARGET" -type f -perm -u+x | head -1)"
[ -x "$BIN" ] || { echo "no binary for $TARGET"; exit 1; }
rm -rf /tmp/reticolo_run && mkdir -p /tmp/reticolo_run && cd /tmp/reticolo_run
echo "+ $BIN $APP_ARGS"
"$BIN" $APP_ARGS
echo "--- run dir ---"; ls -la
REMOTE

OUT="$REPO/tools/runpod/output/$TARGET"; mkdir -p "$OUT"
echo ">> pulling output -> $OUT"
pod_scp -r "root@$RUNPOD_HOST:/tmp/reticolo_run/." "$OUT/" 2>/dev/null || true
ls -la "$OUT"
