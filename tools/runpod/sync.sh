#!/usr/bin/env bash
# rsync the local working tree to the pod (excludes build/, .git/, outputs, h5).
# The other scripts call this first, so a bare `sync.sh` is rarely needed.
#
#   ./tools/runpod/sync.sh
set -euo pipefail
. "$(dirname "$0")/common.sh"

echo ">> rsync $REPO/ -> $RUNPOD_HOST:$RUNPOD_DIR/"
rsync -az --delete \
  -e "ssh -o StrictHostKeyChecking=no -i $RUNPOD_KEY -p $RUNPOD_PORT" \
  --exclude='.git/' --exclude='build/' --exclude='tools/modal/output/' \
  --exclude='tools/kaggle/output/' --exclude='tools/runpod/output/' \
  --exclude='*.h5' --exclude='.DS_Store' \
  "$REPO/" "root@$RUNPOD_HOST:$RUNPOD_DIR/"
echo ">> synced"
