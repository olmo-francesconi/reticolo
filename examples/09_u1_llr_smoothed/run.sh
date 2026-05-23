#!/usr/bin/env bash
# Side-by-side comparison of vanilla LLR vs smoothed LLR (cross-replica
# local-quadratic smoother with summable shrinkage) at the 4D compact
# U(1) bulk transition, beta=1.002, L=6.
#
# Two runs with identical physics + RNG seed; the only difference is
# the driver. Output:
#   results/llr_vanilla.h5     — apps/u1_llr
#   results/llr_smoothed.h5      — apps/u1_llr_smoothed
#
# Run lengths are deliberately modest (n_rm=80) so the comparison
# completes in ~6 min on 8 cores; the goal is to see the transient
# behaviour of <dE>/δ, not to produce a publication-grade ρ(E).

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../_common/preset.sh" "$@"
vanilla_bin="$root/build/$preset/apps/u1_llr"
smoothed_bin="$root/build/$preset/apps/u1_llr_smoothed"

for b in "$vanilla_bin" "$smoothed_bin"; do
    if [[ ! -x $b ]]; then
        echo "binary not found at $b" >&2
        echo "Build first: cmake --build --preset $preset" >&2
        exit 1
    fi
done

results="$here/results"
mkdir -p "$results"
rm -f "$results"/llr_vanilla.h5 "$results"/llr_smoothed.h5

# --- physics (matches examples/05_u1_llr/run_critical.sh) ------------------
ndim=4
size=6
beta=1.002
E_min=2400.0
E_max=3600.0
delta=20.0
tau=1.0
n_md=6
n_nr=2
n_therm_nr=10
n_meas_nr=1000
n_rm=40
n_therm_rm=10
n_meas_rm=1000
exchange=0
seed=42

# --- smoothed-driver knobs ---------------------------------------------------
smooth_K=4
smooth_degree=2
smooth_lambda0=1.0
smooth_lambda_exp=1.0

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}

common_args=(
    --size="$size" --ndim="$ndim" --beta="$beta"
    --E_min="$E_min" --E_max="$E_max" --delta="$delta"
    --tau="$tau" --n_md="$n_md"
    --n_nr="$n_nr" --n_therm_nr="$n_therm_nr" --n_meas_nr="$n_meas_nr"
    --n_rm="$n_rm" --n_therm_rm="$n_therm_rm" --n_meas_rm="$n_meas_rm"
    --exchange="$exchange" --seed="$seed"
)

printf '[%s] vanilla LLR  beta=%s  n_rm=%s  threads=%s\n' \
    "$(date +%H:%M:%S)" "$beta" "$n_rm" "$OMP_NUM_THREADS"
"$vanilla_bin" "${common_args[@]}" --out="$results/llr_vanilla.h5"
printf '[%s] vanilla done\n' "$(date +%H:%M:%S)"

printf '[%s] smoothed  LLR  beta=%s  n_rm=%s  K=%s  deg=%s  λ0=%s  exp=%s\n' \
    "$(date +%H:%M:%S)" "$beta" "$n_rm" "$smooth_K" "$smooth_degree" \
    "$smooth_lambda0" "$smooth_lambda_exp"
"$smoothed_bin" "${common_args[@]}" \
    --smooth_K="$smooth_K" --smooth_degree="$smooth_degree" \
    --smooth_lambda0="$smooth_lambda0" --smooth_lambda_exp="$smooth_lambda_exp" \
    --out="$results/llr_smoothed.h5"
printf '[%s] smoothed done\n' "$(date +%H:%M:%S)"

echo
echo "running analysis: $here/analyze.py"
python3 "$here/analyze.py"

echo
echo "running animation: $here/animate.py"
python3 "$here/animate.py"
