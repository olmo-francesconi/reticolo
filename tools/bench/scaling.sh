#!/usr/bin/env bash
# OpenMP strong- & weak-scaling sweep of the reticolo CPU hot loop.
#
# Measurement kernel is `bench_scaling_all` (per-op wall times for one shape ×
# the ambient OMP_NUM_THREADS); this driver sweeps the thread count and turns
# the times into CSVs consumed by analyze_scaling.py.
#
#   strong (hard): fixed problem size, threads = 1,2,4,…  → speedup t(1)/t(p)
#   weak   (soft): outer dim × p, threads = p             → efficiency t(1)/t(p)
#
# The outer (last) lattice dimension is exactly the axis `exec::partition`
# splits across threads, so growing it by p scales the work-item count by p and
# hands each thread a constant slab — an honest weak-scaling axis.
#
# Local: PRESET=macos-llvm (Apple Clang has no OpenMP). Modal CPU: linux-gcc.
# The linux-nvcc preset has OpenMP OFF — do not use it for CPU scaling.
#
# Usage:
#   PRESET=macos-llvm THREADS="1 2 4" STRONG_SIZES="12 16 20" tools/bench/scaling.sh
# Env knobs (all optional):
#   PRESET        cmake preset to build with          (default macos-llvm)
#   THREADS       thread counts to sweep              (default "1 2 4")
#   STRONG_SIZES  cube edges for the strong sweep     (default "16")   — one curve each
#   WEAK_BASES    per-thread outer-dim bases (4D cube)(default "12")   — one curve each
#   FAMILY        bench family                        (default pair)
#   FORCE_REPS    force/s_full timed reps             (default 20)
#   TRAJ_REPS     trajectory timed reps (0 skips)     (default 8)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"

preset="${PRESET:-macos-llvm}"
threads="${THREADS:-1 2 4}"
strong_sizes="${STRONG_SIZES:-16}"
weak_bases="${WEAK_BASES:-12}"
family="${FAMILY:-pair}"
force_reps="${FORCE_REPS:-20}"
traj_reps="${TRAJ_REPS:-8}"

cmake --build --preset "$preset" --target bench_scaling_all >/dev/null
bin="$root/build/$preset/apps/bench_scaling_all"
[ -x "$bin" ] || { echo "missing $bin" >&2; exit 1; }

results="$here/results"
mkdir -p "$results"
strong_csv="$results/scaling_strong.csv"
weak_csv="$results/scaling_weak.csv"
hdr="mode,base,threads,shape,action,force_ms,sfull_ms,traj_ms"
echo "$hdr" > "$strong_csv"
echo "$hdr" > "$weak_csv"

# run <mode> <base> <shape> <p> <csv>: one bench invocation, rows appended to csv.
# `base` tags the size curve (strong: the cube edge; weak: the per-thread base).
run() {
    local mode="$1" base="$2" shape="$3" p="$4" csv="$5"
    printf '[%s] %-6s base=%-3s p=%-3s shape=%s\n' "$(date +%H:%M:%S)" "$mode" "$base" "$p" "$shape"
    OMP_NUM_THREADS="$p" "$bin" "$shape" "$family" "$force_reps" "$traj_reps" \
        | while read -r action sh th f s t; do
            echo "$mode,$base,$th,$sh,$action,$f,$s,$t" >> "$csv"
        done
}

echo "== strong scaling (sizes: $strong_sizes) =="
for L in $strong_sizes; do
    for p in $threads; do
        run strong "$L" "$L" "$p" "$strong_csv"
    done
done

echo "== weak scaling (bases: $weak_bases; outer dim = base·p) =="
for base in $weak_bases; do
    for p in $threads; do
        outer=$(( base * p ))
        shape="${base}x${base}x${base}x${outer}"
        run weak "$base" "$shape" "$p" "$weak_csv"
    done
done

printf '[%s] wrote %s\n[%s] wrote %s\n' \
    "$(date +%H:%M:%S)" "$strong_csv" "$(date +%H:%M:%S)" "$weak_csv"

if command -v python3 >/dev/null 2>&1; then
    python3 "$here/analyze_scaling.py" "$results"
fi
