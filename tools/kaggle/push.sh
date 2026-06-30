#!/usr/bin/env bash
# Trigger the reticolo CUDA build/test as a headless Kaggle GPU kernel, then
# poll to completion and download the logs — the console-driven GPU CI we can't
# get from Colab's free tier.
#
# Prereqs (one-time):
#   pip install kaggle
#   # create an API token at https://www.kaggle.com/settings -> "Create New Token"
#   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#
# Usage:  tools/kaggle/push.sh
# Env:    KAGGLE_USERNAME  override the username (else read from ~/.kaggle/kaggle.json)
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "${here}/../.." && pwd)"
slug="reticolo-cuda-ci"
ds_slug="reticolo-src"

if ! command -v kaggle >/dev/null 2>&1; then
    echo "error: kaggle CLI not found — 'pip install kaggle' and add ~/.kaggle/kaggle.json" >&2
    exit 1
fi

# Username can come from (in order): env, the classic kaggle.json, or the CLI
# config (the case for an OAuth/access_token setup, which carries no kaggle.json).
detect_user() {
    if [ -n "${KAGGLE_USERNAME:-}" ]; then printf '%s' "${KAGGLE_USERNAME}"; return; fi
    if [ -f "${HOME}/.kaggle/kaggle.json" ]; then
        python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["username"])'
        return
    fi
    kaggle config view 2>/dev/null | sed -n 's/^- username: //p'
}
user="$(detect_user)"
if [ -z "${user}" ]; then
    echo "error: could not determine kaggle username (set KAGGLE_USERNAME, add kaggle.json, or 'kaggle config set -n username -v <you>')" >&2
    exit 1
fi
echo "Kaggle user: ${user}    kernel: ${user}/${slug}    dataset: ${user}/${ds_slug}"

# Ship the LOCAL working tree as a Kaggle dataset (tarball) — so a run reflects
# the current checkout, including uncommitted changes, with no git commit/push.
# Exclude .git, build artifacts, prior Kaggle output, and HDF5 blobs.
dstage="$(mktemp -d)"
trap 'rm -rf "${dstage}"' EXIT
echo "Taring working tree → ${ds_slug}.tar.gz ..."
tar czf "${dstage}/${ds_slug}.tar.gz" -C "${repo}" \
    --exclude='./.git' --exclude='./.git/*' \
    --exclude='./build' --exclude='./build/*' \
    --exclude='./tools/kaggle/output' --exclude='./tools/kaggle/output/*' \
    --exclude='*.h5' --exclude='.DS_Store' .
cat > "${dstage}/dataset-metadata.json" <<EOF
{
  "title": "${ds_slug}",
  "id": "${user}/${ds_slug}",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

# Create the dataset on first run, version it thereafter.
if kaggle datasets status "${user}/${ds_slug}" >/dev/null 2>&1; then
    echo "Versioning dataset ..."
    kaggle datasets version -p "${dstage}" -m "snapshot"
else
    echo "Creating dataset ..."
    kaggle datasets create -p "${dstage}"
fi

# Wait for Kaggle to finish processing the new version before the kernel attaches
# it (an attached-but-unprocessed dataset makes the kernel use a stale version).
echo "Waiting for dataset to process ..."
for _ in $(seq 1 40); do
    st="$(kaggle datasets status "${user}/${ds_slug}" 2>/dev/null || true)"
    echo "  dataset: ${st}"
    case "$(printf '%s' "${st}" | tr '[:upper:]' '[:lower:]')" in
        *ready*|*complete*) break ;;
        *error*) echo "dataset processing error" >&2; exit 1 ;;
    esac
    sleep 10
done

# Stage the kernel: run.py + metadata (GPU + internet on, private, src dataset).
stage="$(mktemp -d)"
trap 'rm -rf "${stage}" "${dstage}"' EXIT
cp "${here}/run.py" "${stage}/run.py"
cat > "${stage}/kernel-metadata.json" <<EOF
{
  "id": "${user}/${slug}",
  "title": "reticolo-cuda-ci",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["${user}/${ds_slug}"],
  "competition_sources": [],
  "kernel_sources": []
}
EOF

echo "Pushing kernel ..."
kaggle kernels push -p "${stage}"

echo "Polling (kernel keeps running if you Ctrl-C) ..."
# Status strings are like "KernelWorkerStatus.RUNNING/COMPLETE/ERROR" — match
# case-insensitively on the terminal states.
while true; do
    status="$(kaggle kernels status "${user}/${slug}" 2>/dev/null || true)"
    echo "  ${status}"
    status_lc="$(printf '%s' "${status}" | tr '[:upper:]' '[:lower:]')"
    case "${status_lc}" in
        *complete*|*error*|*cancel*) break ;;
    esac
    sleep 15
done

out="${here}/output"
mkdir -p "${out}"
# --force: kaggle skips files whose local mtime is newer, which silently keeps a
# STALE throughput.jsonl / kern_*.csv from a prior run; force the fresh artifacts.
kaggle kernels output "${user}/${slug}" -p "${out}" --force
echo "Logs downloaded to ${out} (see *.log)"
case "${status_lc}" in
    *complete*) echo "RESULT: complete" ;;
    *)          echo "RESULT: ${status}" >&2; exit 1 ;;
esac
