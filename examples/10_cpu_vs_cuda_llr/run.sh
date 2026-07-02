#!/usr/bin/env bash
# CPU-vs-CUDA validation of the LLR density of states (phi4, mode A).
#
# The LLR slope a(E_n) ≈ d ln g/dE − 1 is a physical property of the density of
# states, so both backends — the SAME ensemble, different RNG streams (CPU
# xoshiro vs GPU Philox) — must converge to the same a(E_n). We run the SHIPPED
# apps `phi4_llr` (CPU) and `phi4_llr_cuda` (GPU) over the SAME fixed windows for
# several independent seeds, then overlay a(E_n) + reconstructed ln g(E) with a
# cross-seed error band (analyze.py).
#
# The CPU side runs here. The GPU side needs a CUDA device — run it via the Modal
# runner and drop the .h5 files into results/gpu/ (see README.md). This script
# runs CPU, and if results/gpu/ already has runs, it produces the plot.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

preset=${PRESET:-macos-appleclang}
seeds=${SEEDS:-"1 2 3 4 5"}

# Same fixed windows on both backends (bulk of the phi4 action at these couplings,
# probed via phi4_hmc: S ≈ 441 ± 16 for L=8 ndim=3 kappa=0.18 lambda=1).
size=${L:-8}; ndim=${NDIM:-3}; kappa=${KAPPA:-0.18}; lambda=${LAMBDA:-1.0}
e_min=${E_MIN:-400}; e_max=${E_MAX:-480}; delta=${DELTA:-8}
tau=${TAU:-1.0}; n_md=${N_MD:-20}
n_nr=${N_NR:-5}; n_therm_nr=${N_THERM_NR:-200}; n_meas_nr=${N_MEAS_NR:-800}
n_rm=${N_RM:-25}; n_therm_rm=${N_THERM_RM:-100}; n_meas_rm=${N_MEAS_RM:-500}

common="--size=$size --ndim=$ndim --kappa=$kappa --lambda=$lambda \
  --E_min=$e_min --E_max=$e_max --delta=$delta --tau=$tau --n_md=$n_md \
  --n_nr=$n_nr --n_therm_nr=$n_therm_nr --n_meas_nr=$n_meas_nr \
  --n_rm=$n_rm --n_therm_rm=$n_therm_rm --n_meas_rm=$n_meas_rm"

cpu_dir="$here/results/cpu"; gpu_dir="$here/results/gpu"
mkdir -p "$cpu_dir" "$gpu_dir"

echo ">> building phi4_llr (CPU, preset=$preset)"
cmake --build --preset "$preset" --target phi4_llr >/dev/null
llr_bin="$(find "$repo/build/$preset" -name phi4_llr -type f -perm -u+x | head -1)"

echo ">> CPU seeds: $seeds"
for s in $seeds; do
    "$llr_bin" $common --seed="$s" \
        --workspace="$cpu_dir" --out="cpu_seed$s.h5" >/dev/null
    echo "   cpu_seed$s.h5"
done

echo
echo ">> GPU: run phi4_llr_cuda with the SAME args on a CUDA host, e.g. Modal:"
for s in $seeds; do
    echo "   uv run tools/modal/app.py run --app phi4_llr_cuda --name gpu-s$s \\"
    echo "     --args \"$common --seed=$s --out=gpu_seed$s.h5\""
done
echo "   then: pull each run and copy its data/gpu_seed*.h5 into $gpu_dir/"
echo

if compgen -G "$gpu_dir/*.h5" >/dev/null; then
    echo ">> both backends present — plotting"
    uv run --with h5py --with numpy --with matplotlib "$here/analyze.py" \
        --cpu "$cpu_dir"/*.h5 --gpu "$gpu_dir"/*.h5 --out "$here/results/comparison.pdf"
else
    echo ">> no GPU runs in $gpu_dir yet — add them, then re-run to plot."
fi
