#!/usr/bin/env bash
# Per-component HMC breakdown sweep: for each lattice size and thread count, time
# every trajectory pass (refresh / snapshot / kinetic / s_full / fused-kick /
# drift) in isolation and record its effective memory throughput (GB/s), plus a
# STREAM-triad ceiling per thread count. Feeds analyze_breakdown.py.
#
# Measurement kernel is bench_hmc_breakdown. Local: PRESET=macos-llvm. Modal CPU:
# linux-gcc (linux-nvcc has OpenMP OFF).
#
# Per-action size lists: SU3 is ~72× heavier per site and already out of cache at
# any size, while Phi4 must get large (L≳48) to spill L3 — so they sweep
# different edges.
#
# Usage:
#   PRESET=macos-llvm THREADS="1 2 4" PHI4_SIZES="32 48 64" SU3_SIZES="12 16 20" tools/bench/breakdown.sh
# Env knobs (all optional):
#   PRESET      cmake preset                   (default macos-llvm)
#   THREADS     thread counts to sweep         (default "1 2 4")
#   PHI4_SIZES  cube edges for Phi4 (empty=skip)(default "24 32 48 64")
#   SU3_SIZES   cube edges for Wilson<SU3>      (default "12 16 20 24")
#   N_MD        MD steps (sets kick/drift calls-per-traj) (default 8)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"

preset="${PRESET:-macos-llvm}"
threads="${THREADS:-1 2 4}"
phi4_sizes="${PHI4_SIZES:-24 32 48 64}"
su3_sizes="${SU3_SIZES:-12 16 20 24}"
n_md="${N_MD:-8}"

cmake --build --preset "$preset" --target bench_hmc_breakdown >/dev/null
bin="$root/build/$preset/apps/bench_hmc_breakdown"
[ -x "$bin" ] || { echo "missing $bin" >&2; exit 1; }

results="$here/results"
mkdir -p "$results"
csv="$results/breakdown.csv"
echo "base,threads,shape,action,pass,ms,gbps,calls" > "$csv"

echo "== hmc breakdown (phi4: [$phi4_sizes]  su3: [$su3_sizes]  n_md=$n_md) =="
# sweep_action <action> <size-list>
sweep_action() {
    local act="$1" szs="$2"
    for L in $szs; do
        for p in $threads; do
            printf '[%s] %-4s L=%-3s p=%-3s\n' "$(date +%H:%M:%S)" "$act" "$L" "$p"
            OMP_NUM_THREADS="$p" "$bin" "$L" "$act" "$n_md" 2>/dev/null \
                | while read -r a sh th pass ms gbps calls; do
                    echo "$L,$th,$sh,$a,$pass,$ms,$gbps,$calls" >> "$csv"
                done
        done
    done
}
[ -n "$phi4_sizes" ] && sweep_action phi4 "$phi4_sizes"
[ -n "$su3_sizes" ]  && sweep_action su3  "$su3_sizes"

printf '[%s] wrote %s\n' "$(date +%H:%M:%S)" "$csv"
if command -v python3 >/dev/null 2>&1; then
    python3 "$here/analyze_breakdown.py" "$results"
fi
