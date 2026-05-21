#!/usr/bin/env bash
# Two-stage demo of the φ⁴ density of states.
#
# For each κ:
#   1. Run plain HMC. Collect the per-trajectory action S → histogram
#      gives an importance-sampled estimate of ρ(S) in the typical region.
#   2. Read the empirical S range from the HMC samples, pad it, and run
#      LLR over that range. δ (the Gaussian half-width) is the *single*
#      LLR tuning knob: it also sets the replica spacing — the app
#      derives n_rep so that adjacent windows are exactly δ apart.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../_common/preset.sh" "$@"
hmc_bin="$root/build/$preset/apps/phi4_hmc"
llr_bin="$root/build/$preset/apps/phi4_llr"

for b in "$hmc_bin" "$llr_bin"; do
    if [[ ! -x $b ]]; then
        echo "binary not found at $b" >&2
        echo "Build first: cmake --build --preset $preset" >&2
        exit 1
    fi
done

results="$here/results"
mkdir -p "$results"
rm -f "$results"/hmc_kappa*.h5 "$results"/llr_kappa*.h5 "$results"/range_kappa*.txt

ndim=${NDIM:-3}
size=${L:-8}
lambda=${LAMBDA:-1.0}

# HMC for the bulk ρ(S) estimate
n_therm=${N_THERM:-2000}
n_prod=${N_PROD:-20000}

# LLR: delta is the only window knob. n_rep is derived inside the app.
# delta=0 in env => auto-pick s.t. ~20 replicas cover the padded HMC range.
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
target_n_rep=${TARGET_N_REP:-20}

seed=${SEED:-20260518}
jobs=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}

# LLR uses OpenMP internally over replicas. Run llr_jobs parameter values in
# parallel, each with omp_threads threads, so the total ≤ jobs cores.
omp_threads=${OMP_THREADS:-4}
llr_jobs=${LLR_JOBS:-$(( jobs / omp_threads > 0 ? jobs / omp_threads : 1 ))}
export OMP_NUM_THREADS=$omp_threads

# 3D φ⁴ at λ=1, L=8: chi peaks at κ ≈ 0.20 (located by an explicit HMC scan;
# Binder cumulant goes from ~0 at κ=0.16 to its broken-phase value 2/3 by
# κ=0.24). These three κ straddle the transition.
kappas=(${KAPPAS:-0.17 0.20 0.23})

export hmc_bin llr_bin results ndim size lambda
export n_therm n_prod delta tau n_md n_nr n_rm
export n_therm_nr n_meas_nr n_therm_rm n_meas_rm
export range_pad range_lo_pct range_hi_pct target_n_rep seed

stage1_hmc() {
    set -e
    local kappa=$1
    local out="$results/hmc_kappa${kappa}.h5"
    rc=0
    "$hmc_bin" \
        --size="$size" --ndim="$ndim" \
        --kappa="$kappa" --lambda="$lambda" \
        --tau=1.0 --n_md=20 \
        --n_therm="$n_therm" --n_prod="$n_prod" --meas_every=1 \
        --seed="$seed" --out="$out" >/dev/null || rc=$?
    if [[ $rc -ne 0 ]]; then
        printf '[%s] HMC kappa=%s FAILED (exit %d)\n' "$(date +%H:%M:%S)" "$kappa" "$rc" >&2
        return "$rc"
    fi
    printf '[%s] HMC kappa=%s done\n' "$(date +%H:%M:%S)" "$kappa"
}
export -f stage1_hmc

stage2_llr() {
    set -e
    local kappa=$1
    local hmc_out="$results/hmc_kappa${kappa}.h5"
    local rng_out="$results/range_kappa${kappa}.txt"
    local llr_out="$results/llr_kappa${kappa}.h5"

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
    rc=0
    "$llr_bin" \
        --size="$size" --ndim="$ndim" \
        --kappa="$kappa" --lambda="$lambda" \
        --E_min="$e_min" --E_max="$e_max" --delta="$d" \
        --tau="$tau" --n_md="$n_md" \
        --n_nr="$n_nr" --n_therm_nr="$n_therm_nr" --n_meas_nr="$n_meas_nr" \
        --n_rm="$n_rm" --n_therm_rm="$n_therm_rm" --n_meas_rm="$n_meas_rm" \
        --seed="$seed" --out="$llr_out" >/dev/null || rc=$?
    if [[ $rc -ne 0 ]]; then
        printf '[%s] LLR kappa=%s FAILED (exit %d)  range=[%.2f, %.2f]  delta=%.3f\n' \
            "$(date +%H:%M:%S)" "$kappa" "$rc" "$e_min" "$e_max" "$d" >&2
        return "$rc"
    fi
    printf '[%s] LLR kappa=%s done   range=[%.2f, %.2f]  delta=%.3f\n' \
        "$(date +%H:%M:%S)" "$kappa" "$e_min" "$e_max" "$d"
}
export -f stage2_llr

echo "stage 1: HMC for ${#kappas[@]} kappas ($jobs at a time)"
printf '%s\n' "${kappas[@]}" | xargs -L 1 -P "$jobs" bash -c 'stage1_hmc "$@"' _

echo
echo "stage 2: LLR for ${#kappas[@]} kappas ($llr_jobs at a time, $omp_threads threads each)"
printf '%s\n' "${kappas[@]}" | xargs -L 1 -P "$llr_jobs" bash -c 'stage2_llr "$@"' _

echo
echo "running analysis: $here/analyze.py"
python3 "$here/analyze.py"
