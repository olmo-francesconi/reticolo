#!/usr/bin/env bash
# Nsight profiling sweep of the CUDA backend (phi4 + su3 over lattice sizes).
# Runs on a GPU host; writes all artifacts to the output dir for download and
# analysis elsewhere (tools/profile/analyze.py on the Mac, or the Nsight GUIs):
#
#   tools.txt            GPU + nsys/ncu versions (capability probe)
#   throughput.jsonl     one JSON line per (action,L): ms/traj, traj/s, eff GB/s
#   nsys_<a>_L<L>.nsys-rep   Nsight Systems timeline (open in the GUI)
#   kern_<a>_L<L>.csv        per-kernel time summary (nsys stats, CSV)
#   ncu_<a>_L<L>.{ncu-rep,csv,err}   Nsight Compute force-kernel metrics (if perms allow)
#
# `nsys` is trace-based and needs no perf-counter permissions, so it is the
# reliable core. `ncu` reads GPU perf counters, which managed clouds often lock
# (ERR_NVGPUCTRPERM); it is attempted and its failure is non-fatal.
#
# Usage: tools/cuda_profile.sh [build_dir] [out_dir]
set -uo pipefail  # NOT -e: a blocked ncu / nsys must not abort the sweep

build_dir="${1:-build/cuda}"
out="${2:-build/cuda/profile}"
arch="${RETICOLO_CUDA_ARCH:-native}"
phi4_sizes="${RETICOLO_PHI4_SIZES:-8 16 24 32 48 64}"
su3_sizes="${RETICOLO_SU3_SIZES:-8 12 16 24 32}"

export PATH="/usr/local/cuda/bin:${PATH}"
mkdir -p "${out}"

# Resolve nsys; install nsight-systems-cli if absent (trace-based, so it works
# without the GPU perf-counter permission that ncu needs). Try PATH, then known
# install dirs, then apt (direct, then via the NVIDIA CUDA repo keyring).
locate_nsys() {
    command -v nsys 2>/dev/null && return 0
    local f
    f="$(ls /opt/nvidia/nsight-systems*/*/target-linux-x64/nsys \
           /opt/nvidia/nsight-systems*/*/host-linux-x64/nsys \
           /usr/local/cuda*/bin/nsys 2>/dev/null | head -1)"
    [ -z "${f}" ] && f="$(find /opt /usr/local -maxdepth 6 -name nsys -type f 2>/dev/null | head -1)"
    [ -n "${f}" ] && echo "${f}"
}
NSYS="$(locate_nsys)"
if [ -z "${NSYS}" ]; then
    # Minimal CLI-only install per the Nsight Systems Installation Guide. The
    # `nsight-systems-cli` package lives in the NVIDIA *devtools* repo (NOT the
    # cuda repo), keyed on the Ubuntu release (e.g. ubuntu2204/amd64).
    echo "nsys absent — installing nsight-systems-cli from the NVIDIA devtools repo ..."
    apt-get -qq install -y --no-install-recommends gnupg >/dev/null 2>&1 || true
    rel="$( . /etc/lsb-release 2>/dev/null; echo "${DISTRIB_RELEASE:-22.04}" | tr -d . )"
    arch="$(dpkg --print-architecture 2>/dev/null || echo amd64)"
    echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu${rel}/${arch} /" \
        > /etc/apt/sources.list.d/nvidia-devtools.list
    apt-key adv --fetch-keys \
        http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
        >/dev/null 2>&1 || true
    apt-get -qq update || true
    apt-get -qq install -y nsight-systems-cli 2>&1 | tail -3 || echo "nsight-systems-cli install failed"
    NSYS="$(locate_nsys)"
fi
if [ -z "${NSYS}" ]; then
    # Fallback: a versioned nsight-systems package may live in the already-
    # configured CUDA repo (e.g. nsight-systems-2025.1.1) under a different name.
    pkg="$(apt-cache search nsight-systems 2>/dev/null | awk '{print $1}' \
           | grep -iE '^nsight-systems' | sort -V | tail -1)"
    [ -n "${pkg}" ] && { echo "fallback: ${pkg}"; apt-get -qq install -y "${pkg}" >/dev/null 2>&1; }
    NSYS="$(locate_nsys)"
fi
[ -n "${NSYS}" ] && echo "nsys = ${NSYS}"

{
    echo "=== GPU ==="
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
    echo "=== nsys ==="
    if [ -n "${NSYS}" ]; then "${NSYS}" --version 2>&1 | head -3; else echo "nsys MISSING"; fi
    echo "=== ncu ==="; ncu --version 2>&1 | head -3 || echo "ncu MISSING"
} | tee "${out}/tools.txt"

set -e  # configure/build must succeed
cmake -S . -B "${build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DRETICOLO_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${arch}" \
    -DRETICOLO_BUILD_IO=OFF -DRETICOLO_BUILD_APPS=OFF \
    -DRETICOLO_BUILD_TESTS=OFF -DRETICOLO_BUILD_EXAMPLES=OFF \
    -DRETICOLO_ENABLE_OPENMP=OFF
cmake --build "${build_dir}" --target profile_cuda_hmc -j "$(nproc 2>/dev/null || echo 2)"
set +e
bin="${build_dir}/src/cuda/profile_cuda_hmc"

throughput="${out}/throughput.jsonl"
: > "${throughput}"

profile_one() {  # action L
    local action=$1 L=$2 tag="${1}_L${2}"
    echo "--- ${tag} ---"
    "${bin}" --action="${action}" --size="${L}" | tee -a "${throughput}"
    [ -z "${NSYS}" ] && return 0
    # --sample/--cpuctxsw=none: skip host-side CPU sampling, which the managed GPU
    # host blocks (locked perf_event_paranoid) — it only adds a spurious "Unable to
    # configure CPU sampling" error to the report; the CUDA trace is what we want.
    if "${NSYS}" profile --force-overwrite=true --cuda-graph-trace=node --trace=cuda \
           --sample=none --cpuctxsw=none \
           -o "${out}/nsys_${tag}" "${bin}" --action="${action}" --size="${L}" >/dev/null 2>&1; then
        "${NSYS}" stats --report cuda_gpu_kern_sum --format csv --force-export=true \
            "${out}/nsys_${tag}.nsys-rep" > "${out}/kern_${tag}.csv" 2>/dev/null
        "${NSYS}" stats --report cuda_gpu_mem_time_sum --format csv \
            "${out}/nsys_${tag}.nsys-rep" > "${out}/mem_${tag}.csv" 2>/dev/null
    else
        echo "  nsys failed for ${tag}"
    fi
}

for L in ${phi4_sizes}; do profile_one phi4 "${L}"; done
for L in ${su3_sizes}; do profile_one su3 "${L}"; done

# ncu deep dive on the force kernel (eager force-only mode → clean single-kernel
# target). A couple of representative configs; non-fatal if perf counters are
# locked. --launch-count limits ncu's expensive per-kernel replay.
ncu_one() {  # action L
    local action=$1 L=$2 tag="${1}_L${2}"
    echo "--- ncu ${tag} ---"
    "${bin}" --action="${action}" --size="${L}" --force-only | tee -a "${throughput}"
    if ncu --target-processes all --launch-count 3 --set basic \
           -o "${out}/ncu_${tag}" --force-overwrite \
           "${bin}" --action="${action}" --size="${L}" --force-only --iters=3 \
           > "${out}/ncu_${tag}.log" 2> "${out}/ncu_${tag}.err"; then
        ncu --import "${out}/ncu_${tag}.ncu-rep" --csv --page raw \
            > "${out}/ncu_${tag}.csv" 2>/dev/null
        echo "  ncu OK ${tag}"
    else
        echo "  ncu blocked/failed ${tag} (see ncu_${tag}.err — likely ERR_NVGPUCTRPERM)"
    fi
}
ncu_one phi4 32
ncu_one su3 16

echo "=== profile artifacts (${out}) ==="
ls -la "${out}"