# 02 — HMC integrator tuning for φ⁴: which integrator + (τ, n_md) gives the most independent samples per second?

End-to-end comparison of the three HMC integrators on the 3D φ⁴ scalar
field, scanning the (τ, n_md) plane of each one and measuring how fast they
decorrelate the observables in **actual wall-clock time** — the only metric
that matters when you're deciding which one to run.

| integrator       | parameter sweep                            |
|------------------|--------------------------------------------|
| HMC – Leapfrog   | (τ × n_md) = 6 × 10 = 60 runs              |
| HMC – Omelyan2   | (τ × n_md) = 6 × 10 = 60 runs              |
| HMC – Omelyan4   | (τ × n_md) = 6 × 10 = 60 runs              |

180 runs total sharing one HMC-thermalised start state. Phi4 at (ndim=3,
L=16, λ=1.0) across an `easy` (κ=0.12) and a `critical` (κ=0.184) scenario —
the latter near κ_c so critical slowing down clearly separates the integrators.

## Why wall time, not "steps"

An HMC trajectory costs one or more lattice-wide force evaluations, and that
cost scales with `n_md` — so comparing integrators in "trajectories" is
meaningless: Omelyan4 does 4× the force work per step of Leapfrog. The
integrated autocorrelation time `τ_int` in trajectory units would tell you
which integrator gives the most independent samples *per trajectory*; what
you actually care about is which one gives the most independent samples *per
second*. So `tune_phi4` stamps `/prod@wall_seconds` next to the time series,
and `analyze.py` reports

```
τ_int [s] = τ_int [trajectories] × (wall_seconds / n_meas)
```

The bar chart `summary_bar.png` shows `1 / (2·τ_int [s])` — the steady-state
rate at which each integrator decorrelates an observable, in independent
samples per second.

## What this proves

- The three pluggable integrators (Leapfrog, Omelyan2, Omelyan4) are
  selected purely as a type parameter on one `HasFusedKick` HMC chain, so a
  single ~150-LOC C++ binary per integrator drives the same observable record.
- The autocorrelation utility `examples/_common/autocorr.py` (FFT + Madras-
  Sokal windowing) is small (~110 LOC) and reusable from any future example.
- The "right" integrator depends on what you want to measure: Σφ² and S have
  different sensitivities to the dynamics, and the optimum on the
  (τ, n_md) plane shifts with the observable.

## Building & running

Standalone build:

```sh
cd examples/02_phi4_algorithm_tuning
cmake -S . -B build && cmake --build build
```

Or let `run.sh` do it automatically:

```sh
bash   examples/02_phi4_algorithm_tuning/run.sh
python examples/02_phi4_algorithm_tuning/analyze.py
python examples/02_phi4_algorithm_tuning/plot.py
```

## Run it

Configurable via environment variables on `run.sh`: `L`, `NDIM`, `KAPPA`,
`LAMBDA`, `N_THERM`, `N_PROD`, `SEED`, `JOBS`.

## What you get

In `examples/02_phi4_algorithm_tuning/`:

| file | content |
|------|---------|
| `results/<scenario>/<algo>_<params>.h5` | one HDF5 per run (S, Σφ² time series + wall time) |
| `results/<scenario>/summary.csv` | flat table: `algo, τ, n_md, accept, τ_int (units + seconds)` |
| `rho_semilog.pdf` | ρ(t) on a semi-log axis, best run per integrator, per scenario |
| `hmc_step_scan.pdf` | `τ_int(Σφ²)` and acceptance vs step size ε = τ/n_md, per integrator |
| `pareto.pdf` | cost-per-independent-sample vs acceptance, with the per-integrator Pareto frontier |
| `summary_bar.pdf` | best per integrator: independent samples per wall-second |

Read these in roughly this order: step scan → Pareto → ρ(t) overlays
→ summary bar.

## A note on validity

Per-run `τ_int` is reported with the Madras-Sokal statistical uncertainty
`σ(τ_int) ≈ τ_int·√(2(2W+1)/N)` where `W` is the automatic window. With
N = 5000 measurements and typical `τ_int ~ 20-100` updates, the relative
error is ~5-10%, which is fine for the ratios this example highlights but
would be tight for sub-percent claims. `analyze.py` prints a table on
stdout; eyeball the `window_*` columns to confirm convergence (the window
should be a few × τ_int, not the full series).
