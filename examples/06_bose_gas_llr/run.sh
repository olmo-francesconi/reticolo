#!/usr/bin/env bash
# 4D relativistic Bose gas at finite chemical potential — phase-quenched HMC
# diagnostic + LLR reconstruction of ρ(S_I).
#
# Two-stage pipeline per μ, mirroring examples 04 and 05:
#   1. Phase-quenched HMC sampling exp(−S_R(μ)). Records the time series of
#      S_I per trajectory; the empirical (lo, hi) percentiles of that series
#      bracket the LLR window range.
#   2. LLR over [E_min, E_max] with auto-δ to give ~30 replicas. Reconstructs
#      ln ρ(S_I) in the phase-quenched ensemble.
#
# This stage is the LLR-validation step (analyze.py overlays HMC histogram
# and LLR reconstruction). The phase-factor / overlap-free-energy story sits
# downstream on top of the reconstructed ρ(S_I).

set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../.." && pwd)
preset=${RETICOLO_PRESET:-macos-appleclang}
hmc_bin="$root/build/$preset/apps/bose_gas_hmc"
llr_bin="$root/build/$preset/apps/bose_gas_llr"

for b in "$hmc_bin" "$llr_bin"; do
    if [[ ! -x $b ]]; then
        echo "binary not found at $b" >&2
        echo "Build first: cmake --build --preset $preset" >&2
        exit 1
    fi
done

results="$here/results"
mkdir -p "$results"
rm -f "$results"/hmc_mu*.h5 "$results"/llr_mu*.h5 "$results"/range_mu*.txt

ndim=${NDIM:-4}
size=${L:-6}
mass=${MASS:-1.0}
lambda_=${LAMBDA:-1.0}

# HMC stage knobs.
n_therm=${N_THERM:-2000}
n_prod=${N_PROD:-15000}

# LLR knobs.
n_md=${N_MD:-10}
n_nr=${N_NR:-6}
n_rm=${N_RM:-20}
n_therm_nr=${N_THERM_NR:-200}
n_meas_nr=${N_MEAS_NR:-800}
n_therm_rm=${N_THERM_RM:-100}
n_meas_rm=${N_MEAS_RM:-400}
# S_I distribution is symmetric: ρ(s) = ρ(−s). The LLR window grid runs on
# [0, S_I_max) — fixed in intensive units S_I/V — and is scaled to extensive
# S_I via V = size^ndim. This decouples the LLR coverage from whatever the
# HMC pre-stage happens to visit; the HMC stage now exists only to give the
# diagnostic histogram for the comparison plot.
si_over_v_max=${SI_OVER_V_MAX:-0.08}
target_n_rep=${TARGET_N_REP:-30}

seed=${SEED:-20260518}
jobs=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}

# Five μ values bracketing the silver-blaze region for λ = m = 1.
mus=(${MUS:-0.6 0.9 1.1 1.3 1.6})

export hmc_bin llr_bin results ndim size mass lambda_
export n_therm n_prod
export n_md n_nr n_rm n_therm_nr n_meas_nr n_therm_rm n_meas_rm
export si_over_v_max target_n_rep seed

stage1_hmc() {
    local mu=$1
    local out="$results/hmc_mu${mu}.h5"
    "$hmc_bin" \
        -L "$size" --ndim="$ndim" \
        --mass="$mass" --lambda="$lambda_" --mu="$mu" \
        --tau=1.0 --n_md=10 \
        --n_therm="$n_therm" --n_prod="$n_prod" --meas_every=1 \
        --seed="$seed" --out="$out" >/dev/null
    printf '[%s] HMC mu=%s done\n' "$(date +%H:%M:%S)" "$mu"
}
export -f stage1_hmc

stage2_llr() {
    local mu=$1
    local rng_out="$results/range_mu${mu}.txt"
    local llr_out="$results/llr_mu${mu}.h5"

    # Fixed range in intensive S_I/V; extensive e_max = (S_I/V)_max · V.
    # delta = per-replica spacing referenced to the FULL symmetric range
    # [-e_max, +e_max] so the LLR coverage on [0, e_max] has the same
    # per-window width it would have in the symmetric case.
    python3 - "$rng_out" <<PY
import os, sys
volume       = int(os.environ['size']) ** int(os.environ['ndim'])
si_over_v    = float(os.environ['si_over_v_max'])
target_n_rep = int(os.environ['target_n_rep'])
s_max = si_over_v * volume
# delta = per-replica spacing referenced to the full symmetric range
# [-s_max, +s_max]. Windows on [0, s_max] are shifted by delta/2 so the
# innermost window centre sits at S_I = +delta/2, NOT at S_I = 0 — by
# symmetry the slope at the natural peak is trivially 0 anyway, so we
# spend the replicas where they actually probe non-trivial structure.
delta = (2.0 * s_max) / (target_n_rep - 1)
e_min = delta / 2.0
e_max = s_max - delta / 2.0
with open(sys.argv[1], 'w') as out:
    out.write(f"{e_min} {e_max} {delta}\n")
PY
    read -r e_min e_max d <"$rng_out"
    "$llr_bin" \
        -L "$size" --ndim="$ndim" \
        --mass="$mass" --lambda="$lambda_" --mu="$mu" \
        --E_min="$e_min" --E_max="$e_max" --delta="$d" \
        --tau=1.0 --n_md="$n_md" \
        --n_nr="$n_nr" --n_therm_nr="$n_therm_nr" --n_meas_nr="$n_meas_nr" \
        --n_rm="$n_rm" --n_therm_rm="$n_therm_rm" --n_meas_rm="$n_meas_rm" \
        --seed="$seed" --out="$llr_out" >/dev/null
    printf '[%s] LLR mu=%s done   range=[%.2f, %.2f]  delta=%.3f\n' \
        "$(date +%H:%M:%S)" "$mu" "$e_min" "$e_max" "$d"
}
export -f stage2_llr

echo "stage 1: HMC for ${#mus[@]} mu values ($jobs at a time)"
printf '%s\n' "${mus[@]}" | xargs -L 1 -P "$jobs" bash -c 'stage1_hmc "$@"' _

echo
echo "stage 2: LLR for ${#mus[@]} mu values ($jobs at a time)"
printf '%s\n' "${mus[@]}" | xargs -L 1 -P "$jobs" bash -c 'stage2_llr "$@"' _

echo
echo "Done. Now run:  python3 $here/analyze.py"
