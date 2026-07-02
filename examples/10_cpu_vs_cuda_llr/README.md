# 10 — CPU vs CUDA: LLR density of states agreement

A cross-backend **physics validation**: the CPU and CUDA backends run the SAME
LLR simulation and must agree on a physical observable, not merely compile.

## What it validates

The LLR slope `a(E_n) ≈ d ln g/dE − 1` is a property of the density of states
`g(E)`, independent of the sampler. The two backends are the **same ensemble but
different Markov chains** — CPU xoshiro vs GPU Philox RNG — so a correct port must
converge to the same `a(E_n)` within statistical error, even though no single
trajectory matches.

Two models, selected with `MODEL`:

- `MODEL=phi4` (default) — `phi4_llr` / `phi4_llr_cuda`, **mode A** (the window
  constrains the real action `S`).
- `MODEL=bose` — `bose_gas_llr` / `bose_gas_llr_cuda`, **mode B** (the window
  constrains the imaginary action `S_I` of the relativistic Bose gas at finite
  µ). This directly validates the complex-LLR GPU port.

It runs the shipped apps over the **same fixed windows** for several independent
seeds, then overlays:

1. the per-seed `a(E_n)` **point-clouds** of both backends,
2. the reconstructed `ln g(E)` (trapezoidal integral of the cross-seed mean `a`),
3. the per-window two-sample **KS p-value** (distribution match).

The two backends' `a(E_n)` distributions being statistically indistinguishable
across the well-sampled range is the pass condition.

## Method: compare distributions, not point estimates

- **Per-seed estimator = the LAST Robbins-Monro `a`-value** (the converged RM
  iterate), not the mean over the tail. A tail-mean smears in the RM transient
  and the strong within-run autocorrelation of the RM tail.
- At the final iteration the replica exchange is a permutation, so each seed
  contributes exactly one final `a` per window. Per window we then have a CPU
  sample and a GPU sample (one value per seed each), and we test whether they are
  drawn from the **same distribution** (two-sample Kolmogorov–Smirnov) — the
  physical claim is that the two backends are the same ensemble, so per-window
  their converged-`a` distributions must match, not merely their means. Use ≥5
  seeds per backend for the test to have any resolution.

`tools/validate/compare_llr.py` is the numeric companion (prints a per-window
table without plotting).

## Tuning: more intervals ≠ smaller δ

To add windows, shrink `--spacing` (window-centre spacing), **not** `--delta`
(window width). `δ` sets LLR stability: a too-narrow window makes the NR/RM
update + replica exchange blow up stochastically (a runaway `a` at a window,
then smeared across neighbours by param-swap exchange) — on *both* backends, so
it corrupts the comparison. Keep `δ` in the stable range (δ≈2 for the bose `S_I`
here, whose spread is ≈2.65) and place centres closer with `SPACING=1` for more,
overlapping intervals. Also keep window centres inside the well-sampled range
(≲2σ of the observable); deeper tails are hard to warm into and converge slowly.

## Running

The CPU side runs locally; the GPU side needs a CUDA device (there is no local
GPU here — it goes through the Modal runner).

```sh
# 1. CPU seeds (local) + prints the exact GPU commands. MODEL=phi4|bose.
MODEL=phi4 SEEDS="1 2 3 4 5" ./run.sh

# 2. GPU seeds (Modal), for each seed printed by run.sh:
uv run tools/modal/app.py run --app phi4_llr_cuda --name gpu-phi4-s1 \
    --args "<same args> --seed=1 --out=gpu_seed1.h5"
uv run tools/modal/app.py pull <run_id>
cp tools/modal/output/<run_id>/**/gpu_seed1.h5 results/phi4/gpu/

# 3. re-run to plot once results/<model>/gpu/ is populated:
MODEL=phi4 ./run.sh          # or call analyze.py directly
```

`analyze.py` can be pointed at any two run sets:

```sh
uv run --with h5py --with numpy --with scipy --with matplotlib ./analyze.py \
    --cpu results/phi4/cpu/*.h5 --gpu results/phi4/gpu/*.h5 \
    --title "phi4 LLR — CPU vs CUDA" --out results/comparison_phi4.pdf
```

## Extending

`MODEL=bose` already validates the complex-LLR (mode B) GPU port: `a(S_I)` is
**odd** in `S_I` (the S_I distribution is even, so `d ln ρ/dS_I` is odd), and both
backends must reproduce that antisymmetry. Any other action pair with matching
CPU/GPU LLR apps and the same output schema drops into `run.sh`'s `case` the same
way.
