#!/usr/bin/env bash
# Generalized-observable LLR: reconstruct the density of states of the field
# amplitude Q = Σφ² (NOT the action) for the φ⁴ scalar field.
#
#   1. Plain HMC — collect ⟨φ²⟩ per trajectory. Its histogram is an
#      importance-sampled ρ(⟨φ²⟩), accurate only in the typical band.
#   2. Read the empirical ⟨φ²⟩ range, pad it, and run LLR windowing on the
#      SAME observable via orch::llr::Orchestrator. δ (per-site, ⟨φ²⟩ units) is the
#      single knob — it also sets the replica spacing, so the app derives
#      n_rep to cover the range with adjacent windows δ apart.
#   3. analyze.py integrates the converged a_n to ln ρ(⟨φ²⟩) and overlays
#      the direct histogram in its overlap band.
#
# The only thing that makes this "generalized" vs example 03 is the observable
# the LLR windows on — a one-line Constraint swap in the driver (observable.hpp).

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../_common/preset.sh" "$@"
build_example
hmc_bin="$example_bin/phi4_hmc"
llr_bin="$example_bin/generalized_dos_llr"

results="$here/results"
mkdir -p "$results"
rm -f "$results"/hmc.h5 "$results"/amp_llr.h5 "$results"/range.txt

ndim=${NDIM:-3}
size=${L:-8}
kappa=${KAPPA:-0.18}
lambda=${LAMBDA:-1.0}
seed=${SEED:-42}

# HMC for the bulk ρ(⟨φ²⟩) estimate
n_therm=${N_THERM:-2000}
n_prod=${N_PROD:-20000}

# LLR window knobs (per-site ⟨φ²⟩ units). The range is fixed WIDE on purpose —
# well beyond the band plain HMC explores — so the reconstruction reaches deep
# into the suppressed tails. The direct histogram then validates the overlap.
amp_min=${AMP_MIN:-0.38}
amp_max=${AMP_MAX:-0.88}
damp=${DAMP:-0.017}      # window δ = replica spacing (the single LLR knob)
tau=${TAU:-1.0}
n_md=${N_MD:-20}
n_nr=${N_NR:-6}
n_rm=${N_RM:-18}
n_therm_nr=${N_THERM_NR:-80}
n_meas_nr=${N_MEAS_NR:-250}
n_therm_rm=${N_THERM_RM:-50}
n_meas_rm=${N_MEAS_RM:-250}

omp_threads=${OMP_THREADS:-4}
export OMP_NUM_THREADS=$omp_threads

echo "stage 1: HMC (κ=$kappa, ${ndim}D L=$size) → direct ⟨φ²⟩ histogram"
"$hmc_bin" \
    --size="$size" --ndim="$ndim" \
    --kappa="$kappa" --lambda="$lambda" \
    --tau=1.0 --n_md=20 \
    --n_therm="$n_therm" --n_prod="$n_prod" --meas_every=1 \
    --seed="$seed" --workspace="$results" --out="hmc.h5" >/dev/null

echo
echo "stage 2: generalized LLR windowing on Q=Σφ² over ⟨φ²⟩ ∈ [$amp_min, $amp_max], δ=$damp"
"$llr_bin" \
    --size="$size" --ndim="$ndim" \
    --kappa="$kappa" --lambda="$lambda" \
    --amp_min="$amp_min" --amp_max="$amp_max" --damp="$damp" \
    --tau="$tau" --n_md="$n_md" \
    --n_nr="$n_nr" --n_therm_nr="$n_therm_nr" --n_meas_nr="$n_meas_nr" \
    --n_rm="$n_rm" --n_therm_rm="$n_therm_rm" --n_meas_rm="$n_meas_rm" \
    --seed="$seed" --workspace="$results" --out="amp_llr.h5"

echo
echo "running analysis: $here/analyze.py"
python3 "$here/analyze.py"
