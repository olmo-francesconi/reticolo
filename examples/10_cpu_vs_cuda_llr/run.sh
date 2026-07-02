#!/usr/bin/env bash
# CPU-vs-CUDA validation of the LLR density of states. Runs the SHIPPED apps for
# one MODEL on both backends over the SAME fixed windows for several independent
# seeds, then overlays a(E_n) + reconstructed ln g with a cross-seed error band
# (analyze.py). Both backends are the same ensemble, different RNG streams (CPU
# xoshiro vs GPU Philox), so a correct port must agree on a(E_n).
#
#   MODEL=phi4  (default)  phi4_llr / phi4_llr_cuda  — mode A, constrains action S
#   MODEL=bose             bose_gas_llr / bose_gas_llr_cuda — mode B, constrains S_I
#
# The CPU side runs here. The GPU side needs a CUDA device — run it via the Modal
# runner and drop the .h5 files into results/<model>/gpu/ (see README.md). This
# script runs CPU, prints the GPU commands, and plots once GPU runs are present.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

preset=${PRESET:-macos-appleclang}
seeds=${SEEDS:-"1 2 3 4 5"}
model=${MODEL:-phi4}

# Same LLR schedule for both models; the couplings + windows differ per model.
tau=${TAU:-1.0}; n_md=${N_MD:-20}
n_nr=${N_NR:-5}; n_therm_nr=${N_THERM_NR:-200}; n_meas_nr=${N_MEAS_NR:-800}
n_rm=${N_RM:-25}; n_therm_rm=${N_THERM_RM:-100}; n_meas_rm=${N_MEAS_RM:-500}
sched="--tau=$tau --n_md=$n_md --n_nr=$n_nr --n_therm_nr=$n_therm_nr \
  --n_meas_nr=$n_meas_nr --n_rm=$n_rm --n_therm_rm=$n_therm_rm --n_meas_rm=$n_meas_rm"

case "$model" in
  phi4)
    cpu_app=phi4_llr; gpu_app=phi4_llr_cuda
    # bulk of the phi4 action (probed via phi4_hmc: S ≈ 441 ± 16, L=8 ndim=3).
    phys="--size=${L:-8} --ndim=${NDIM:-3} --kappa=${KAPPA:-0.18} --lambda=${LAMBDA:-1.0} \
      --E_min=${E_MIN:-400} --E_max=${E_MAX:-480} --delta=${DELTA:-8}"
    n_md=${N_MD:-20}
    title="phi4 LLR — CPU vs CUDA"
    xlabel='$E_n$  (action window centre)'
    ;;
  bose)
    cpu_app=bose_gas_llr; gpu_app=bose_gas_llr_cuda
    # S_I is symmetric ~N(0, 2.65) at these couplings (probed via bose_gas_hmc,
    # L=4 ndim=4 mu=1); windows span the bulk + moderate tails.
    phys="--size=${L:-4} --ndim=${NDIM:-4} --mass=${MASS:-1.0} --lambda=${LAMBDA:-1.0} \
      --mu=${MU:-1.0} --E_min=${E_MIN:--8} --E_max=${E_MAX:-8} --delta=${DELTA:-2}"
    title="Bose gas LLR (mode B) — CPU vs CUDA"
    xlabel='$E_n$  ($S_I$ window centre)'
    ;;
  *) echo "unknown MODEL=$model (phi4|bose)"; exit 2 ;;
esac
common="$phys $sched"

cpu_dir="$here/results/$model/cpu"; gpu_dir="$here/results/$model/gpu"
mkdir -p "$cpu_dir" "$gpu_dir"

echo ">> [$model] building $cpu_app (CPU, preset=$preset)"
cmake --build --preset "$preset" --target "$cpu_app" >/dev/null
llr_bin="$(find "$repo/build/$preset" -name "$cpu_app" -type f -perm -u+x | head -1)"

echo ">> [$model] CPU seeds: $seeds"
for s in $seeds; do
    "$llr_bin" $common --seed="$s" --workspace="$cpu_dir" --out="cpu_seed$s.h5" >/dev/null
    echo "   cpu_seed$s.h5"
done

echo
echo ">> [$model] GPU: run $gpu_app with the SAME args on a CUDA host, e.g. Modal:"
for s in $seeds; do
    echo "   uv run tools/modal/app.py run --app $gpu_app --name gpu-$model-s$s \\"
    echo "     --args \"$common --seed=$s --out=gpu_seed$s.h5\""
done
echo "   then pull each run and copy its gpu_seed*.h5 into $gpu_dir/"
echo

if compgen -G "$gpu_dir/*.h5" >/dev/null; then
    echo ">> both backends present — plotting"
    uv run --with h5py --with numpy --with scipy --with matplotlib "$here/analyze.py" \
        --cpu "$cpu_dir"/*.h5 --gpu "$gpu_dir"/*.h5 \
        --title "$title" --xlabel "$xlabel" \
        --out "$here/results/comparison_$model.pdf"
else
    echo ">> no GPU runs in $gpu_dir yet — add them, then re-run to plot."
fi
