#!/usr/bin/env bash
# Two-stage demo of the density of states for 4D SU(2) Wilson gauge theory.
#
# For each beta:
#   1. Run plain SU(2) HMC (`su2_hmc`). The per-trajectory action S goes
#      into a histogram — an importance-sampled estimate of rho(S) in the
#      typical region only.
#   2. Read the empirical S range, pad, and run SU(2) LLR (`su2_llr`) over
#      that range. delta is the single LLR tuning knob (Gaussian half-width
#      = replica spacing); n_rep is derived in the app.
#
# SU(2) in 4D has no sharp bulk transition — it's a smooth crossover. The
# point of the demo is that LLR reconstructs rho(S) across many orders of
# magnitude, including the suppressed tails where the HMC histogram has no
# signal. Three beta values in the intermediate/weak-coupling regime give
# clearly distinct rho(S) curves.

set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../.." && pwd)
preset=${RETICOLO_PRESET:-macos-appleclang}
hmc_bin="$root/build/$preset/apps/su2_hmc"
llr_bin="$root/build/$preset/apps/su2_llr"

for b in "$hmc_bin" "$llr_bin"; do
    if [[ ! -x $b ]]; then
        echo "binary not found at $b" >&2
        echo "Build first: cmake --build --preset $preset" >&2
        exit 1
    fi
done

results="$here/results"
mkdir -p "$results"
rm -f "$results"/hmc_beta*.h5 "$results"/llr_beta*.h5 "$results"/range_beta*.txt

ndim=${NDIM:-4}
size=${L:-4}

n_therm=${N_THERM:-1000}
n_prod=${N_PROD:-8000}

delta=${DELTA:-0}
tau=${TAU:-1.0}
n_md=${N_MD:-20}
n_nr=${N_NR:-6}
n_rm=${N_RM:-30}
n_therm_nr=${N_THERM_NR:-200}
n_meas_nr=${N_MEAS_NR:-800}
n_therm_rm=${N_THERM_RM:-100}
n_meas_rm=${N_MEAS_RM:-400}
range_pad=${RANGE_PAD:-0.25}
range_lo_pct=${RANGE_LO_PCT:-0.5}
range_hi_pct=${RANGE_HI_PCT:-99.5}
target_n_rep=${TARGET_N_REP:-30}

seed=${SEED:-20260520}
jobs=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}

omp_threads=${OMP_THREADS:-4}
llr_jobs=${LLR_JOBS:-$(( jobs / omp_threads > 0 ? jobs / omp_threads : 1 ))}
export OMP_NUM_THREADS=$omp_threads

# Three beta values in the intermediate-coupling regime of 4D SU(2). The
# upper bound is set by LLR stability: at β ≳ 2.5 on L = 4 the natural HMC
# S-distribution narrows enough that the auto-tuned δ falls into a regime
# where the LLR Gaussian-penalty force scale (S − E_n)/δ² becomes large
# and the inner-replica HMC integrator goes unstable. Raise TARGET_N_REP
# down or pin DELTA explicitly to push beyond β ~ 2.4 on this volume.
betas=(${BETAS:-2.0 2.2 2.4})

export hmc_bin llr_bin results ndim size
export n_therm n_prod delta tau n_md n_nr n_rm
export n_therm_nr n_meas_nr n_therm_rm n_meas_rm
export range_pad range_lo_pct range_hi_pct target_n_rep seed

stage1_hmc() {
    local beta=$1
    local out="$results/hmc_beta${beta}.h5"
    "$hmc_bin" \
        --size="$size" --ndim="$ndim" --beta="$beta" \
        --tau=1.0 --n_md=20 \
        --n_therm="$n_therm" --n_prod="$n_prod" --meas_every=1 \
        --seed="$seed" --out="$out" >/dev/null
    printf '[%s] HMC beta=%s done\n' "$(date +%H:%M:%S)" "$beta"
}
export -f stage1_hmc

stage2_llr() {
    local beta=$1
    local hmc_out="$results/hmc_beta${beta}.h5"
    local rng_out="$results/range_beta${beta}.txt"
    local llr_out="$results/llr_beta${beta}.h5"

    python3 - "$hmc_out" "$rng_out" <<PY
import sys, h5py, numpy as np, os
lo_pct       = float(os.environ['range_lo_pct'])
hi_pct       = float(os.environ['range_hi_pct'])
pad          = float(os.environ['range_pad'])
target_n_rep = int(os.environ['target_n_rep'])
delta_env    = float(os.environ['delta'])
with h5py.File(sys.argv[1], 'r') as f:
    s = f['/prod/obs/s'][:]
lo, hi = np.percentile(s, [lo_pct, hi_pct])
span = hi - lo
e_min = lo - pad * span
e_max = hi + pad * span
delta = delta_env if delta_env > 0 else (e_max - e_min) / (target_n_rep - 1)
with open(sys.argv[2], 'w') as out:
    out.write(f"{e_min} {e_max} {delta}\n")
PY
    read -r e_min e_max d <"$rng_out"
    "$llr_bin" \
        --size="$size" --ndim="$ndim" --beta="$beta" \
        --E_min="$e_min" --E_max="$e_max" --delta="$d" \
        --tau="$tau" --n_md="$n_md" \
        --n_nr="$n_nr" --n_therm_nr="$n_therm_nr" --n_meas_nr="$n_meas_nr" \
        --n_rm="$n_rm" --n_therm_rm="$n_therm_rm" --n_meas_rm="$n_meas_rm" \
        --seed="$seed" --out="$llr_out" >/dev/null
    printf '[%s] LLR beta=%s done   range=[%.2f, %.2f]  delta=%.3f\n' \
        "$(date +%H:%M:%S)" "$beta" "$e_min" "$e_max" "$d"
}
export -f stage2_llr

echo "stage 1: HMC for ${#betas[@]} betas ($jobs at a time)"
printf '%s\n' "${betas[@]}" | xargs -L 1 -P "$jobs" bash -c 'stage1_hmc "$@"' _

echo
echo "stage 2: LLR for ${#betas[@]} betas ($llr_jobs at a time, $omp_threads threads each)"
printf '%s\n' "${betas[@]}" | xargs -L 1 -P "$llr_jobs" bash -c 'stage2_llr "$@"' _

echo
echo "Done. Now run:  python3 $here/analyze.py"
