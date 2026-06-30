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

source "$(dirname "${BASH_SOURCE[0]}")/../_common/preset.sh" "$@"
build_example
hmc_bin="$example_bin/bose_gas_hmc"
llr_bin="$example_bin/bose_gas_llr"

results="$here/results"
mkdir -p "$results"
rm -f "$results"/hmc_mu*.h5 "$results"/llr_mu*.h5 "$results"/range_mu*.txt

ndim=${NDIM:-4}
size=${L:-4}
mass=${MASS:-1.0}
lambda_=${LAMBDA:-1.0}

# HMC stage knobs. n_prod is set below, matched to the LLR budget.
n_therm=${N_THERM:-2000}

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
target_n_rep=${TARGET_N_REP:-60}
# Window width as a multiple of the centre spacing. 1 = non-overlapping
# (width == spacing); >1 keeps wide, soft windows on a denser grid — the
# window force stiffens as 1/width^2, so this decouples constraint
# resolution from integrator stability (n_md must scale as 1/width if the
# width is reduced instead).
overlap=${OVERLAP:-2.0}
# Independent LLR repeats per mu (seed, seed+1, ...) — the spread of the
# per-seed fit+integral pipeline is the statistical error on <e^{iS_I}>.
llr_seeds=${LLR_SEEDS:-4}

# Apples-to-apples HMC budget: production trajectories = the total LLR
# measurement count per mu — windows (~target_n_rep/2 on the half range)
# × (NR + RM measures per window) × seeds.
llr_total_meas=$(( target_n_rep / 2 * (n_nr * n_meas_nr + n_rm * n_meas_rm) * llr_seeds ))
n_prod=${N_PROD:-$llr_total_meas}

seed=${SEED:-42}
jobs=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}

# LLR uses OpenMP internally over replicas. Run llr_jobs parameter values in
# parallel, each with omp_threads threads, so the total ≤ jobs cores.
omp_threads=${OMP_THREADS:-2}
llr_jobs=${LLR_JOBS:-$(( jobs / omp_threads > 0 ? jobs / omp_threads : 1 ))}
export OMP_NUM_THREADS=$omp_threads

# Full sweep across the silver-blaze transition for λ = m = 1
# (μ_c = 1.165(7) at 4^4, PRD 101, 014504).
mus=(${MUS:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0})

export hmc_bin llr_bin results ndim size mass lambda_
export n_therm n_prod
export n_md n_nr n_rm n_therm_nr n_meas_nr n_therm_rm n_meas_rm
export si_over_v_max target_n_rep overlap seed

stage1_hmc() {
    set -e
    local mu=$1
    local out="hmc_mu${mu}.h5"
    rc=0
    "$hmc_bin" \
        -L "$size" --ndim="$ndim" \
        --mass="$mass" --lambda="$lambda_" --mu="$mu" \
        --tau=1.0 --n_md=10 \
        --n_therm="$n_therm" --n_prod="$n_prod" --meas_every=1 \
        --seed="$seed" --workspace="$results" --out="$out" >/dev/null || rc=$?
    if [[ $rc -ne 0 ]]; then
        printf '[%s] HMC mu=%s FAILED (exit %d)\n' "$(date +%H:%M:%S)" "$mu" "$rc" >&2
        return "$rc"
    fi
    printf '[%s] HMC mu=%s done\n' "$(date +%H:%M:%S)" "$mu"
}
export -f stage1_hmc

stage2_llr() {
    set -e
    local mu=$1 sd=$2
    local rng_out="$results/range_mu${mu}_s${sd}.txt"
    local llr_out="llr_mu${mu}_s${sd}.h5"

    # Fixed range in intensive S_I/V; extensive e_max = (S_I/V)_max · V.
    # delta = per-replica spacing referenced to the FULL symmetric range
    # [-e_max, +e_max] so the LLR coverage on [0, e_max] has the same
    # per-window width it would have in the symmetric case.
    python3 - "$rng_out" <<PY
import os, sys
volume       = int(os.environ['size']) ** int(os.environ['ndim'])
si_over_v    = float(os.environ['si_over_v_max'])
target_n_rep = int(os.environ['target_n_rep'])
overlap      = float(os.environ['overlap'])
s_max = si_over_v * volume
# spacing = per-replica centre spacing referenced to the full symmetric
# range [-s_max, +s_max]. Windows on [0, s_max] are shifted by spacing/2
# so the innermost centre sits at S_I = +spacing/2, NOT at S_I = 0 — by
# symmetry the slope at the natural peak is trivially 0 anyway, so we
# spend the replicas where they actually probe non-trivial structure.
# Window width = overlap * spacing (overlapping when overlap > 1).
spacing = (2.0 * s_max) / (target_n_rep - 1)
delta   = overlap * spacing
e_min = spacing / 2.0
e_max = s_max - spacing / 2.0
with open(sys.argv[1], 'w') as out:
    out.write(f"{e_min} {e_max} {spacing} {delta}\n")
PY
    read -r e_min e_max sp d <"$rng_out"
    rc=0
    "$llr_bin" \
        -L "$size" --ndim="$ndim" \
        --mass="$mass" --lambda="$lambda_" --mu="$mu" \
        --E_min="$e_min" --E_max="$e_max" --spacing="$sp" --delta="$d" \
        --tau=1.0 --n_md="$n_md" \
        --n_nr="$n_nr" --n_therm_nr="$n_therm_nr" --n_meas_nr="$n_meas_nr" \
        --n_rm="$n_rm" --n_therm_rm="$n_therm_rm" --n_meas_rm="$n_meas_rm" \
        --seed="$sd" --workspace="$results" --out="$llr_out" >/dev/null || rc=$?
    if [[ $rc -ne 0 ]]; then
        printf '[%s] LLR mu=%s seed=%s FAILED (exit %d)  range=[%.2f, %.2f]  delta=%.3f\n' \
            "$(date +%H:%M:%S)" "$mu" "$sd" "$rc" "$e_min" "$e_max" "$d" >&2
        return "$rc"
    fi
    printf '[%s] LLR mu=%s seed=%s done   range=[%.2f, %.2f]  spacing=%.3f  delta=%.3f\n' \
        "$(date +%H:%M:%S)" "$mu" "$sd" "$e_min" "$e_max" "$sp" "$d"
}
export -f stage2_llr

echo "stage 1: HMC for ${#mus[@]} mu values, n_prod=$n_prod ($jobs at a time)"
printf '%s\n' "${mus[@]}" | xargs -L 1 -P "$jobs" bash -c 'stage1_hmc "$@"' _

combos=()
for mu in "${mus[@]}"; do
    for ((k = 0; k < llr_seeds; k++)); do combos+=("$mu $((seed + k))"); done
done

echo
echo "stage 2: LLR, ${#mus[@]} mu x $llr_seeds seeds ($llr_jobs at a time, $omp_threads threads each)"
printf '%s\n' "${combos[@]}" | xargs -L 1 -P "$llr_jobs" bash -c 'stage2_llr "$@"' _

echo
echo "running analysis: $here/analyze.py"
python3 "$here/analyze.py"
