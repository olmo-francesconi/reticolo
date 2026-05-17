#!/usr/bin/env bash
# Sweep kappa across the phi^4 phase transition (lambda fixed) and produce one
# HDF5 file per point. Pair with analyze.py for the susceptibility plot.

set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../.." && pwd)
preset=${RETICOLO_PRESET:-macos-appleclang}
binary="$root/build/$preset/apps/phi4_hmc"

if [[ ! -x $binary ]]; then
    echo "phi4_hmc binary not found at $binary" >&2
    echo "Build it first: cmake --build --preset $preset" >&2
    exit 1
fi

results="$here/results"
mkdir -p "$results"
rm -f "$results"/phi4_kappa_*.h5

L=${L:-6}
lambda=${LAMBDA:-0.02}
n_therm=${N_THERM:-200}
n_prod=${N_PROD:-2000}
seed=${SEED:-20260517}

# Kappa range chosen to bracket the broken-symmetry transition for these
# values of lambda and L; widen if tuning a different point in the (kappa,
# lambda) plane.
kappas=(0.115 0.120 0.124 0.127 0.129 0.131 0.133 0.135 0.138 0.142 0.150)

for kappa in "${kappas[@]}"; do
    out="$results/phi4_kappa_${kappa}.h5"
    echo "==> kappa=$kappa  -> $out"
    "$binary" \
        --size="$L" --kappa="$kappa" --lambda="$lambda" \
        --n_therm="$n_therm" --n_prod="$n_prod" --seed="$seed" \
        --out="$out"
done

echo
echo "Done. Now run:  python3 $here/analyze.py"
