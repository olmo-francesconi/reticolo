#!/usr/bin/env bash
# Sweep kappa across the 2D phi^4 broken-symmetry transition for several
# lattice sizes; pair with analyze.py for the finite-size-scaling collapse.
# 2D Ising universality gives exact analytic exponents (nu = 1, gamma/nu = 7/4)
# that the collapse fit should reproduce to within a couple of percent.
#
# Runs `JOBS` sweeps in parallel (default: one per core).

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../_common/preset.sh" "$@"
binary="$root/build/$preset/apps/phi4_hmc"

if [[ ! -x $binary ]]; then
    echo "phi4_hmc binary not found at $binary" >&2
    echo "Build it first: cmake --build --preset $preset" >&2
    exit 1
fi

results="$here/results"
mkdir -p "$results"
rm -f "$results"/phi4_L*_kappa*.h5

ndim=${NDIM:-2}
lambda=${LAMBDA:-1.145}
n_therm=${N_THERM:-3000}
n_prod=${N_PROD:-50000}
seed=${SEED:-20260517}
jobs=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}

# Four sizes spanning V = 256 -> 2304 give a clean lever arm in 2D; kappa grid
# bracketed densely around the peak (kappa_c ≈ 0.33 for lambda = 1.145).
sizes=(${SIZES:-16 24 32 48})
kappas=(0.300 0.315 0.322 0.328 0.330 0.331 0.332 0.333 0.334 0.335 0.336 0.337 0.339 0.342 0.348 0.358 0.370)

export binary results ndim lambda n_therm n_prod seed

run_one() {
    set -e
    local L=$1 kappa=$2
    local out="$results/phi4_L${L}_kappa${kappa}.h5"
    rc=0
    "$binary" \
        --size="$L" --kappa="$kappa" --lambda="$lambda" --ndim="$ndim" \
        --n_therm="$n_therm" --n_prod="$n_prod" --seed="$seed" \
        --out="$out" >/dev/null || rc=$?
    if [[ $rc -ne 0 ]]; then
        printf '[%s] L=%-3s kappa=%s  FAILED (exit %d)\n' "$(date +%H:%M:%S)" "$L" "$kappa" "$rc" >&2
        return "$rc"
    fi
    printf '[%s] L=%-3s kappa=%s  done\n' "$(date +%H:%M:%S)" "$L" "$kappa"
}
export -f run_one

n_jobs=$(( ${#sizes[@]} * ${#kappas[@]} ))
echo "running $n_jobs sweeps ($jobs at a time)"

{
    for L in "${sizes[@]}"; do
        for kappa in "${kappas[@]}"; do
            printf '%s %s\n' "$L" "$kappa"
        done
    done
} | xargs -L 1 -P "$jobs" bash -c 'run_one "$@"' _

echo
echo "running analysis: $here/analyze.py"
python3 "$here/analyze.py"
