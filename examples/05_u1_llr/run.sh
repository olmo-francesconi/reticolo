#!/usr/bin/env bash
# Two-stage demo of the density of states for 4D compact U(1) gauge theory.
#
# For each beta:
#   1. Run plain gauge HMC. The per-trajectory action S goes into a
#      histogram — an importance-sampled estimate of rho(S) in the typical
#      region only.
#   2. Read the empirical S range, pad, and run gauge LLR over that range.
#      delta is the single LLR tuning knob (Gaussian half-width = replica
#      spacing); n_rep is derived in the app.
#
# Literature target: Langfeld, Lucini, Rago 2012 (EPJC) showed a clear
# double-peak rho(S) at the weakly-first-order transition near beta ~ 1.01
# in 4D U(1). Default L = 4 here is small for finite-size purposes — at
# L = 4 the transition is a crossover — but the LLR curves still clearly
# show the broadening / asymmetry signature near criticality.

set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../.." && pwd)
preset=${RETICOLO_PRESET:-macos-appleclang}
hmc_bin="$root/build/$preset/apps/u1_hmc"
llr_bin="$root/build/$preset/apps/u1_llr"

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

n_therm=${N_THERM:-2000}
n_prod=${N_PROD:-15000}

delta=${DELTA:-0}
tau=${TAU:-1.0}
n_md=${N_MD:-20}
n_nr=${N_NR:-6}
n_rm=${N_RM:-40}
n_therm_nr=${N_THERM_NR:-200}
n_meas_nr=${N_MEAS_NR:-1000}
n_therm_rm=${N_THERM_RM:-100}
n_meas_rm=${N_MEAS_RM:-500}
range_pad=${RANGE_PAD:-0.30}
range_lo_pct=${RANGE_LO_PCT:-0.5}
range_hi_pct=${RANGE_HI_PCT:-99.5}
target_n_rep=${TARGET_N_REP:-30}

seed=${SEED:-20260518}
jobs=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}

# Three betas straddling the 4D compact U(1) transition (bulk ~1.01;
# at L=4 the crossover is a bit smeared).
betas=(${BETAS:-0.95 1.00 1.05})

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
echo "stage 2: LLR for ${#betas[@]} betas ($jobs at a time)"
printf '%s\n' "${betas[@]}" | xargs -L 1 -P "$jobs" bash -c 'stage2_llr "$@"' _

echo
echo "Done. Now run:  python3 $here/analyze.py"
