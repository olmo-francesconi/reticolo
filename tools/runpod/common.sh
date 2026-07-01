# Shared config + helpers for the RunPod SSH profiling workflow. Sourced by the
# other scripts here; not run directly.
#
# Configure the connection once via env vars — easiest is a gitignored
# tools/runpod/env.local (copy env.local.example) that this file sources:
#   RUNPOD_HOST   pod IP            (required)
#   RUNPOD_PORT   pod SSH port      (required)
#   RUNPOD_KEY    private key path  (default ~/.ssh/runpod)
#   RUNPOD_ARCH   CUDA arch sm_XX   (V100=70, A100=80, H100=90; default 70)
#   RUNPOD_DIR    remote source dir (default /root/reticolo)

set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$_here/../.." && pwd)"
[ -f "$_here/env.local" ] && . "$_here/env.local"

: "${RUNPOD_HOST:?set RUNPOD_HOST/PORT (see tools/runpod/README.md)}"
: "${RUNPOD_PORT:?set RUNPOD_HOST/PORT (see tools/runpod/README.md)}"
: "${RUNPOD_KEY:=$HOME/.ssh/runpod}"
: "${RUNPOD_ARCH:=70}"
: "${RUNPOD_DIR:=/root/reticolo}"

_ssh_opts=(-o StrictHostKeyChecking=no -o ConnectTimeout=20 -i "$RUNPOD_KEY" -p "$RUNPOD_PORT")

# Run a command on the pod.
pod()     { ssh "${_ssh_opts[@]}" "root@$RUNPOD_HOST" "$@"; }
# scp helper (note: scp uses -P for the port).
pod_scp() { scp -o StrictHostKeyChecking=no -i "$RUNPOD_KEY" -P "$RUNPOD_PORT" "$@"; }

# Run a heredoc script (on stdin) on the pod with NAME=value vars in its env.
# ssh joins its command args with spaces and the REMOTE shell re-splits them, so
# a space-containing value passed positionally to `bash -s` (CMAKE_ARGS, an app's
# flag list) gets corrupted — only the first word survives. Passing such values
# as shell-quoted env assignments prefixed before `bash -s` keeps them intact;
# the heredoc reads them as $NAME.
#   pod_run DIR="$RUNPOD_DIR" ARGS="$CMAKE_ARGS" <<'REMOTE' ... REMOTE
pod_run() {
    local pre=""
    for kv in "$@"; do
        pre+="${kv%%=*}=$(printf '%q' "${kv#*=}") "
    done
    pod "${pre}bash -s"
}

# Remote shell preamble: put the REAL CUDA toolkit on PATH. The /usr/local/cuda
# symlink can get repointed to a config-only stub by apt, so pick the newest
# cuda-*/bin that actually has nvcc. Also add /usr/local/bin for the pip cmake.
REMOTE_ENV='export PATH="$(for d in $(ls -d /usr/local/cuda-*/bin 2>/dev/null | sort -Vr) /usr/local/cuda/bin; do [ -x "$d/nvcc" ] && { echo "$d"; break; }; done):/usr/local/bin:$PATH"'

# CMake flags that make the linux-nvcc preset build on the pod: the target arch,
# g++-13 as host compiler (Ubuntu 22.04's default GCC 11 lacks <format>), and
# -allow-unsupported-compiler so nvcc accepts a slightly-newer GCC.
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=$RUNPOD_ARCH -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=g++-13 -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler"
