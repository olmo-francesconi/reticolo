# 03 — φ⁴ density of states: HMC histogram vs LLR reconstruction

Two-stage end-to-end demo of the `llr/` module on the real-valued φ⁴
scalar action.

## What this example does

For each κ in a small grid around the broken-symmetry transition:

1. **Stage 1 — plain HMC.** Runs `phi4_hmc` and stores `S` per
   trajectory. A histogram of that chain gives an importance-sampled
   estimate of ρ(S) — accurate where ρ is large, *empty* in the tails.
2. **Range selection.** Reads the HMC `S` chain, takes its central
   percentile range, pads it on both sides, and uses that to set the
   LLR window `[E_min, E_max]`. δ is chosen so adjacent windows
   overlap.
3. **Stage 2 — LLR.** Runs `phi4_llr` over the chosen window using
   Newton-Raphson warm-up + restarted Robbins-Monro + alternating
   even/odd nearest-neighbour replica exchange. Outputs the per-replica
   slopes `a_n`.
4. **Comparison.** `analyze.py` reconstructs `ln ρ(S)` by trapezoidal
   integration of `a(E)` along the replica grid, aligns it to the HMC
   histogram at the HMC peak, and overlays the two per κ. They should
   agree where they overlap; the LLR curve should extend smoothly into
   the suppressed-S region the HMC chain doesn't reach.

## Why two stages

Plain HMC concentrates on the typical-S region — that's the importance
sampler doing its job, but it's also why ρ(S) in the tails costs
exponential statistics to estimate. LLR factorises the problem: each
replica is biased to live in its window, and `a_n` directly estimates
the local slope `d ln ρ / dS`. Integrating those slopes reconstructs ρ
over the *full* chosen range at constant relative precision.

The HMC stage is therefore a check + a useful range-finder, not the
endpoint.

## Building & running

Standalone build:

```sh
cd examples/03_phi4_llr
cmake -S . -B build && cmake --build build
```

Or let `run.sh` do it automatically:

```sh
bash examples/03_phi4_llr/run.sh
python3 examples/03_phi4_llr/analyze.py
```

Override the preset with `RETICOLO_PRESET=macos-llvm ./run.sh` (recommended on
macOS for OpenMP-enabled LLR replicas).

## Run it

Outputs in `examples/03_phi4_llr/results/`:
* `hmc_kappa<k>.h5`  — full HMC output (per-trajectory S, magnetisation, etc.)
* `range_kappa<k>.txt` — `E_min E_max delta` chosen for the LLR run
* `llr_kappa<k>.h5`  — LLR slopes per replica + exchange stats

And `rho_hmc_vs_llr.png` next to the script: one subplot per κ, HMC
histogram (markers) and aligned LLR `ln ρ(S)` (line).

## What this is *not*

A literature replication. The published non-gauge LLR papers use
discrete spins (Potts, q-state) or complex-action scalars (Bose gas at
finite μ). Plain real-action φ⁴ doesn't appear in the LLR literature
we found, so this example is a *sanity demonstration* of the
implementation, with the HMC histogram as the cross-check.

## Knobs (env vars)

| variable          | default              | meaning                                       |
| ----------------- | -------------------- | --------------------------------------------- |
| `RETICOLO_PRESET` | `macos-appleclang`   | which build preset to invoke                  |
| `JOBS`            | `nproc`              | parallel runs at once                         |
| `NDIM`            | `3`                  | spatial dimensions                            |
| `L`               | `8`                  | linear lattice extent                         |
| `LAMBDA`          | `1.0`                | quartic coupling                              |
| `KAPPAS`          | `0.17 0.20 0.23`     | space-separated κ list (straddle κ_c ≈ 0.20)  |
| `N_THERM`         | `2000`               | HMC thermalisation trajectories               |
| `N_PROD`          | `20000`              | HMC production trajectories                   |
| `DELTA`           | `0` (auto)           | Gaussian half-width = replica spacing (the single LLR knob); 0 ⇒ pick s.t. ~`TARGET_N_REP` windows fit the padded HMC range |
| `TARGET_N_REP`    | `20`                 | replica count used by the auto-δ heuristic    |
| `RANGE_PAD`       | `0.30`               | fractional padding added on each side         |
| `RANGE_LO_PCT`    | `0.5`                | lower percentile of HMC S used as range low   |
| `RANGE_HI_PCT`    | `99.5`               | upper percentile of HMC S used as range high  |
| `N_NR`, `N_RM`    | `6`, `40`            | NR warm-up / RM sweep counts                  |
| `N_THERM_NR`, `N_MEAS_NR` | `200`, `1000` | HMC traj counts inside each NR iter           |
| `N_THERM_RM`, `N_MEAS_RM` | `100`, `500`  | HMC traj counts inside each RM sweep          |
| `TAU`, `N_MD`     | `1.0`, `20`          | HMC trajectory length / MD steps              |
| `SEED`            | `20260518`           | RNG seed (offset per κ within run.sh)         |

The defaults are sized for a quick demo run on a laptop. For sharper
tails, raise `N_RM` and lower `DELTA`.
