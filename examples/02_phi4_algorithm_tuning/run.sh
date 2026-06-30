#!/usr/bin/env bash
# Parameter sweep for the HMC integrator-tuning example.
#
# Single shared HMC thermalisation lands the lattice into a typical
# equilibrium configuration; every production run then loads that state and
# runs straight into measurement with its own integrator + (τ, n_md). That
# keeps the comparison fair — no integrator is penalised by its own warm-up.
#
# Grid:
#   HMC (×3 integrators)  τ ∈ {0.25, 0.5, 1, 2, 4, 8} × n_md ∈ {1, 2, 3, 4, 6, 8, 12, 16, 20, 24}
# Total: 3·60 = 180 production runs + 1 thermalisation.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../_common/preset.sh" "$@"
build_example
bindir="$example_bin"

scenario=${SCENARIO:-easy}
results="$here/results/$scenario"
mkdir -p "$results"
rm -f "$results"/*.h5 "$results"/thermalized.bin

ndim=${NDIM:-3}
L=${L:-16}
# Default coupling per scenario. The two scenarios bracket the 3D φ⁴ Ising
# transition (κ_c ≈ 0.184 for λ=1.0): 'easy' sits deep in the symmetric phase
# where autocorrelations are short and all algorithms are competitive;
# 'critical' sits near κ_c where critical slowing down separates them.
if [[ -z ${KAPPA:-} ]]; then
    case "$scenario" in
        easy)     kappa=0.12  ;;
        critical) kappa=0.184 ;;
        *) kappa=0.18 ;;
    esac
else
    kappa=$KAPPA
fi
lambda=${LAMBDA:-1.0}
n_therm_shared=${N_THERM_SHARED:-5000}
n_prod=${N_PROD:-20000}
seed=${SEED:-42}
jobs=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}
state_path="$results/thermalized.bin"

echo "scenario=$scenario  kappa=$kappa  L=$L  n_prod=$n_prod  jobs=$jobs"
echo

# ---- shared HMC thermalisation (Omelyan2, tau=1.0, n_md=10) ----
echo "[$(date +%H:%M:%S)] thermalising scenario=$scenario (kappa=$kappa, HMC Omelyan2 tau=1.0 n_md=10, $n_therm_shared trajectories)"
"$bindir/tune_phi4_hmc_omelyan2" \
    --size="$L" --kappa="$kappa" --lambda="$lambda" --ndim="$ndim" \
    --tau=1.0 --n_md=10 \
    --n_therm="$n_therm_shared" --n_prod=0 \
    --seed="$seed" \
    --out="$results/_thermalize.h5" --save_state="$state_path" \
    >/dev/null
rm -f "$results/_thermalize.h5"
echo "[$(date +%H:%M:%S)] thermalised state ready ($(wc -c <"$state_path") bytes)"
echo

taus=(0.25 0.5 1.0 2.0 4.0 8.0)
n_mds=(1 2 3 4 6 8 12 16 20 24)

export bindir results ndim L kappa lambda n_prod seed state_path

run_hmc() {
    set -e
    local algo=$1 tau=$2 n_md=$3
    local tag=$(printf 'tau%05.2f_nmd%03d' "$tau" "$n_md")
    local out="$results/${algo}_${tag}.h5"
    rc=0
    "$bindir/tune_phi4_${algo}" \
        --size="$L" --kappa="$kappa" --lambda="$lambda" --ndim="$ndim" \
        --tau="$tau" --n_md="$n_md" \
        --n_therm=0 --n_prod="$n_prod" --seed="$seed" \
        --init_from="$state_path" --out="$out" \
        >/dev/null || rc=$?
    if [[ $rc -ne 0 ]]; then
        printf '[%s] %-13s tau=%-5s n_md=%-3s FAILED (exit %d)\n' \
            "$(date +%H:%M:%S)" "$algo" "$tau" "$n_md" "$rc" >&2
        return "$rc"
    fi
    printf '[%s] %-13s tau=%-5s n_md=%-3s done\n' \
        "$(date +%H:%M:%S)" "$algo" "$tau" "$n_md"
}
export -f run_hmc

n_jobs=$(( 3 * ${#taus[@]} * ${#n_mds[@]} ))
echo "running $n_jobs production sweeps ($jobs at a time)"
echo "Phi4: ndim=$ndim L=$L kappa=$kappa lambda=$lambda  (n_prod=$n_prod)"
echo

{
    for algo in hmc_leapfrog hmc_omelyan2 hmc_omelyan4; do
        for tau in "${taus[@]}"; do
            for n_md in "${n_mds[@]}"; do
                printf 'h %s %s %s\n' "$algo" "$tau" "$n_md"
            done
        done
    done
} | xargs -L 1 -P "$jobs" bash -c '
    case "$1" in
        h) run_hmc "$2" "$3" "$4" ;;
    esac
' _

echo
echo "running analysis: $here/analyze.py"
python3 "$here/analyze.py"
echo "running plot:     $here/plot.py"
python3 "$here/plot.py"
