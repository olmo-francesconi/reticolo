# 10 — CPU vs CUDA: LLR density of states agreement

A cross-backend **physics validation**: the CPU and CUDA backends run the SAME
LLR simulation and must agree on a physical observable, not merely compile.

## What it validates

The LLR slope `a(E_n) ≈ d ln g/dE − 1` is a property of the density of states
`g(E)`, independent of the sampler. The two backends are the **same ensemble but
different Markov chains** — CPU xoshiro vs GPU Philox RNG — so a correct port must
converge to the same `a(E_n)` within statistical error, even though no single
trajectory matches.

This runs the shipped apps `phi4_llr` (CPU) and `phi4_llr_cuda` (GPU) over the
**same fixed windows** for several independent seeds, then overlays:

1. `a(E_n)` from both backends with a **cross-seed** error band,
2. the reconstructed `ln g(E)` (trapezoidal integral of `a`),
3. the per-window pull `(a_GPU − a_CPU)/σ` against a ±3σ guide.

Agreement across the well-sampled action range is the pass condition.

## Why cross-seed errors

A single run's `a(E_n)` comes from one Robbins-Monro trajectory whose tail is
strongly autocorrelated — `std/√N` over that tail is **not** a valid error (see
`tools/validate/compare_llr.py`). The only honest error bar is the scatter of the
converged `a` across independent seeds, so pass ≥3–5 seeds per backend.

## Running

The CPU side runs locally; the GPU side needs a CUDA device (there is no local
GPU here — it goes through the Modal runner).

```sh
# 1. CPU seeds (local) + prints the exact GPU commands:
SEEDS="1 2 3 4 5" ./run.sh

# 2. GPU seeds (Modal), for each seed printed by run.sh:
uv run tools/modal/app.py run --app phi4_llr_cuda --name gpu-s1 \
    --args "<same args> --seed=1 --out=gpu_seed1.h5"
uv run tools/modal/app.py pull <run_id>
cp tools/modal/output/<run_id>/**/gpu_seed1.h5 results/gpu/

# 3. re-run to plot once results/gpu/ is populated:
./run.sh          # or call analyze.py directly
```

`analyze.py` can be pointed at any two run sets:

```sh
uv run --with h5py --with numpy --with matplotlib ./analyze.py \
    --cpu results/cpu/*.h5 --gpu results/gpu/*.h5 --out results/comparison.pdf
```

It reuses the multi-seed loader from `tools/validate/compare_llr.py` (that script
is the numeric companion — it prints the per-window Δ/σ table without plotting).

## Extending

`phi4_llr` is mode A (real action). The same harness applies to the complex
`bose_gas_llr` / `bose_gas_llr_cuda` (mode B: the window constrains the imaginary
action `S_I`) — point `analyze.py` at those outputs to validate the complex-LLR
GPU port the same way.
