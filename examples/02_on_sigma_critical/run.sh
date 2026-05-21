#!/usr/bin/env bash
# Sweep beta across the 3D O(3) sigma-model critical region for several lattice
# sizes using the Wolff cluster updater; pair with analyze.py for the Binder
# crossings + collapse.
#
# Runs `JOBS` sweeps in parallel (default: one per core).

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../_common/preset.sh" "$@"
binary="$root/build/$preset/apps/on_sigma_wolff"

if [[ ! -x $binary ]]; then
    echo "on_sigma_wolff binary not found at $binary" >&2
    echo "Build it first: cmake --build --preset $preset" >&2
    exit 1
fi

results="$here/results"
mkdir -p "$results"
rm -f "$results"/on_sigma_*.h5

ndim=${NDIM:-3}
n_cluster=${N_CLUSTER:-5}
n_therm=${N_THERM:-2000}
n_prod=${N_PROD:-30000}
seed=${SEED:-20260517}
jobs=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}

# Sizes and beta grid bracket the 3D O(3) transition (beta_c ≈ 0.693).
# Five sizes spanning V from 216 to 8000 give a clean lever arm for FSS;
# beta grid is dense near beta_c and pads out symmetrically to anchor the
# wings of the Binder curve.
sizes=(${SIZES:-6 8 12 16 20})
betas=(0.660 0.670 0.678 0.684 0.688 0.691 0.692 0.6925 0.693 0.6935 0.694 0.6945 0.695 0.697 0.700 0.705 0.715 0.725)

export binary results ndim n_cluster n_therm n_prod seed

run_one() {
    set -e
    local L=$1 beta=$2
    local out="$results/on_sigma_L${L}_beta${beta}.h5"
    rc=0
    "$binary" \
        --size="$L" --beta="$beta" --ndim="$ndim" \
        --n_cluster="$n_cluster" --n_therm="$n_therm" --n_prod="$n_prod" \
        --seed="$seed" --out="$out" >/dev/null || rc=$?
    if [[ $rc -ne 0 ]]; then
        printf '[%s] L=%-3s beta=%s  FAILED (exit %d)\n' "$(date +%H:%M:%S)" "$L" "$beta" "$rc" >&2
        return "$rc"
    fi
    printf '[%s] L=%-3s beta=%s  done\n' "$(date +%H:%M:%S)" "$L" "$beta"
}
export -f run_one

n_jobs=$(( ${#sizes[@]} * ${#betas[@]} ))
echo "running $n_jobs sweeps ($jobs at a time)"

{
    for L in "${sizes[@]}"; do
        for beta in "${betas[@]}"; do
            printf '%s %s\n' "$L" "$beta"
        done
    done
} | xargs -L 1 -P "$jobs" bash -c 'run_one "$@"' _

echo
echo "running analysis: $here/analyze.py"
python3 "$here/analyze.py"
