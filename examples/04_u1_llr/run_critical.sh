#!/usr/bin/env bash
# Paper-replication run: single LLR sweep at the 4D compact U(1) bulk
# transition to resolve the double-peak structure of rho(S).
#
# beta_c ~= 1.0106(18)  (arxiv hep-lat/0210010, confirmed by
# Langfeld-Lucini-Pellegrini-Rago, EPJC 76:306 (2016) / arxiv 1509.08391).
# L = 6 is the smallest size where the double peak is unambiguous.
#
# No HMC stage. The LLR window, spacing and run lengths are hard-coded
# from a prior calibration so the script is fully reproducible.
#
# Approx wall time on 8 cores (OMP_NUM_THREADS=8): ~15 min.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../_common/preset.sh" "$@"
llr_bin="$root/build/$preset/apps/u1_llr"

if [[ ! -x $llr_bin ]]; then
    echo "binary not found at $llr_bin" >&2
    echo "Build first: cmake --build --preset $preset" >&2
    exit 1
fi

results="$here/results"
mkdir -p "$results"
rm -f "$results"/llr_critical_beta*.h5

# --- physics ---------------------------------------------------------------
ndim=4
size=6
beta=1.002

# --- LLR knobs (fixed) -----------------------------------------------------
E_min=2400.0
E_max=3600.0
delta=20.0
# HMC
tau=1.0
n_md=6
# NR
n_nr=2
n_therm_nr=10
n_meas_nr=1000
# RM
n_rm=240
n_therm_rm=2
n_meas_rm=100
# replica-exchange
exchange=0  # 0: off, 1: on

seed=42
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}

out="$results/llr_critical_beta${beta}.h5"

printf '[%s] LLR beta=%s  range=[%s, %s]  delta=%s  threads=%s\n' \
    "$(date +%H:%M:%S)" "$beta" "$E_min" "$E_max" "$delta" "$OMP_NUM_THREADS"

"$llr_bin" \
    --size="$size" --ndim="$ndim" --beta="$beta" \
    --E_min="$E_min" --E_max="$E_max" --delta="$delta" \
    --tau="$tau" --n_md="$n_md" \
    --n_nr="$n_nr" --n_therm_nr="$n_therm_nr" --n_meas_nr="$n_meas_nr" \
    --n_rm="$n_rm" --n_therm_rm="$n_therm_rm" --n_meas_rm="$n_meas_rm" \
    --exchange="$exchange" \
    --seed="$seed" --out="$out"

printf '[%s] LLR beta=%s done   -> %s\n' "$(date +%H:%M:%S)" "$beta" "$out"

echo
echo "running analysis: $here/analyze_critical.py"
python3 "$here/analyze_critical.py"
