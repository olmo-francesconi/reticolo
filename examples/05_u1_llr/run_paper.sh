#!/usr/bin/env bash
# Paper-replication run for the 4D compact U(1) bulk first-order transition.
#
# Target: reproduce the qualitative double-peak structure of rho(E_p) at the
# transition (beta_c ~ 1.0106(18); arxiv hep-lat/0210010) as shown by
# Langfeld-Lucini-Pellegrini-Rago (arxiv 1509.08391, EPJC 76:306 (2016)).
#
# Lattice size is bumped to L=6 (was L=4 in the quick demo) — at L=4 the
# transition is smeared into a crossover. L=8 would be cleaner but ~5x
# slower; L=6 is the practical sweet spot for a single-machine overnight run.
#
# Two-stage pipeline identical to run.sh: plain HMC -> empirical S range ->
# LLR over that range with auto-delta.
#
# Approx wall time on 8 cores at the defaults (OMP_THREADS=4, two LLR jobs
# in parallel × 4 threads each): ~25 min.

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
size=${L:-6}

# HMC stage knobs — longer than the L=4 demo since we need a confident
# range estimate around the transition.
n_therm=${N_THERM:-4000}
n_prod=${N_PROD:-30000}

# LLR knobs — denser grid (~30 replicas) and more RM sweeps than the demo
# to suppress the 1/(k+1) tail and resolve the double peak cleanly.
delta=${DELTA:-0}
tau=${TAU:-1.0}
n_md=${N_MD:-6}
n_nr=${N_NR:-6}
n_rm=${N_RM:-40}
n_therm_nr=${N_THERM_NR:-300}
n_meas_nr=${N_MEAS_NR:-1200}
n_therm_rm=${N_THERM_RM:-150}
n_meas_rm=${N_MEAS_RM:-500}
range_pad=${RANGE_PAD:-0.30}
range_lo_pct=${RANGE_LO_PCT:-0.5}
range_hi_pct=${RANGE_HI_PCT:-99.5}
target_n_rep=${TARGET_N_REP:-30}

seed=${SEED:-20260518}
jobs=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}

# LLR uses OpenMP internally over replicas. Run llr_jobs parameter values in
# parallel, each with omp_threads threads, so the total ≤ jobs cores.
omp_threads=${OMP_THREADS:-4}
llr_jobs=${LLR_JOBS:-$(( jobs / omp_threads > 0 ? jobs / omp_threads : 1 ))}
export OMP_NUM_THREADS=$omp_threads

# Three betas tightly bracketing beta_c ~= 1.0106:
#   * 0.998  — Coulomb side, dilute monopoles, smooth ρ(S).
#   * 1.010  — at the transition, where the double-peak / shoulder shows up.
#   * 1.020  — confining side, condensed monopoles, smooth ρ(S).
betas=(${BETAS:-0.998 1.010 1.020})

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
