# 03 — Algorithm tuning for φ⁴: which updater + parameters give the most independent samples per second?

End-to-end comparison of four Markov-chain updaters on the 3D φ⁴ scalar
field, scanning the parameter knob of each one and measuring how fast they
decorrelate the observables in **actual wall-clock time** — the only metric
that matters when you're deciding which one to run.

| algorithm        | parameter sweep                |
|------------------|--------------------------------|
| Metropolis       | proposal width σ (8 values)    |
| HMC – Leapfrog   | (τ × n_md) = 4 × 4 = 16 runs   |
| HMC – Omelyan2   | (τ × n_md) = 4 × 4 = 16 runs   |
| HMC – Omelyan4   | (τ × n_md) = 4 × 4 = 16 runs   |

56 runs total, 5000 production measurements each, after 1000 thermalisation
updates. Phi4 at (ndim=3, L=12, κ=0.18, λ=1.0) — near-critical so the
autocorrelations are strong enough to clearly separate algorithms.

## Why wall time, not "steps"

It is meaningless to compare a Metropolis sweep (V site updates, no force
evaluation) to an HMC trajectory (one or more lattice-wide force evaluations)
in "updates" — they cost wildly different amounts of CPU time. The
integrated autocorrelation time `τ_int` in update units would tell you which
algorithm gives the most independent samples *per update*; what you actually
care about is which one gives the most independent samples *per second*. So
`tune_phi4` stamps `/prod@wall_seconds` next to the time series, and
`analyze.py` reports

```
τ_int [s] = τ_int [updates] × (wall_seconds / n_meas)
```

The bar chart `summary_bar.png` shows `1 / (2·τ_int [s])` — the steady-state
rate at which each algorithm decorrelates an observable, in independent
samples per second.

## What this proves

- A `LocalAction` updater wrapper (`alg::Metropolis`) and an `HasFusedKick`
  HMC chain with three pluggable integrators (Leapfrog, Omelyan2, Omelyan4)
  share enough API surface that one ~150-LOC C++ binary can drive all four
  on the same observable record.
- The autocorrelation utility `examples/_common/autocorr.py` (FFT + Madras-
  Sokal windowing) is small (~110 LOC) and reusable from any future example.
- The "right" integrator depends on what you want to measure: Σφ² and S have
  different sensitivities to the dynamics, and the optimum on the
  (τ, n_md) plane shifts with the observable.

## Building & running

Standalone build:

```sh
cd examples/03_phi4_algorithm_tuning
cmake -S . -B build && cmake --build build
```

Or let `run.sh` do it automatically:

```sh
bash   examples/03_phi4_algorithm_tuning/run.sh
python examples/03_phi4_algorithm_tuning/analyze.py
python examples/03_phi4_algorithm_tuning/plot.py
```

## Run it

Configurable via environment variables on `run.sh`: `L`, `NDIM`, `KAPPA`,
`LAMBDA`, `N_THERM`, `N_PROD`, `SEED`, `JOBS`.

## What you get

In `examples/03_phi4_algorithm_tuning/`:

| file | content |
|------|---------|
| `results/<algo>_<params>.h5` | one HDF5 per run (S, Σφ² time series + wall time) |
| `results/summary.csv` | flat table: `algo, params, accept, τ_int (units + seconds)` |
| `autocorr_rho.png` | ρ(t) curves vs lag-in-seconds, best parameter per algorithm |
| `metropolis_tau.png` | `τ_int [s]` vs σ for Metropolis |
| `hmc_heatmaps.png` | 3 panels: `τ_int [s]` on the (τ, n_md) plane per HMC integrator |
| `hmc_surfaces.png` | same data as 3D surfaces (`log₁₀ τ_int [s]`) |
| `summary_bar.png` | best per algorithm: independent samples per wall-second |

Read these in roughly this order: heatmaps → 3D surfaces → ρ(t) overlays
→ summary bar.

## A note on validity

Per-run `τ_int` is reported with the Madras-Sokal statistical uncertainty
`σ(τ_int) ≈ τ_int·√(2(2W+1)/N)` where `W` is the automatic window. With
N = 5000 measurements and typical `τ_int ~ 20-100` updates, the relative
error is ~5-10%, which is fine for the ratios this example highlights but
would be tight for sub-percent claims. `analyze.py` prints a table on
stdout; eyeball the `window_*` columns to confirm convergence (the window
should be a few × τ_int, not the full series).
