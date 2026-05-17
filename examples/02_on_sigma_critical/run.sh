#!/usr/bin/env bash
# Sweep beta across the 3D O(3) sigma-model critical region for several lattice
# sizes; pair with analyze.py to plot the Binder cumulant crossings.

set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../.." && pwd)
preset=${RETICOLO_PRESET:-macos-appleclang}
binary="$root/build/$preset/apps/on_sigma_metropolis"

if [[ ! -x $binary ]]; then
    echo "on_sigma_metropolis binary not found at $binary" >&2
    echo "Build it first: cmake --build --preset $preset" >&2
    exit 1
fi

results="$here/results"
mkdir -p "$results"
rm -f "$results"/on_sigma_*.h5

ndim=${NDIM:-3}
n_therm=${N_THERM:-500}
n_prod=${N_PROD:-3000}
seed=${SEED:-20260517}

# Sizes and beta grid chosen to bracket the 3D O(3) transition (beta_c ≈ 0.693).
sizes=(${SIZES:-4 6 8})
betas=(0.62 0.65 0.67 0.69 0.71 0.73 0.76)

for L in "${sizes[@]}"; do
    for beta in "${betas[@]}"; do
        out="$results/on_sigma_L${L}_beta${beta}.h5"
        echo "==> L=$L beta=$beta  -> $out"
        "$binary" \
            --size="$L" --beta="$beta" --ndim="$ndim" \
            --n_therm="$n_therm" --n_prod="$n_prod" --seed="$seed" \
            --out="$out"
    done
done

echo
echo "Done. Now run:  python3 $here/analyze.py"
