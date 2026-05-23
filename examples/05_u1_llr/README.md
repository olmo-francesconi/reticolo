# 05 — Compact U(1) gauge density of states with LLR + replica exchange

End-to-end demo of the gauge stack (`gauge/`) on 4D compact U(1) Wilson
gauge theory. Mirrors example 04 (φ⁴ scalar) but on link fields.

## What this example does

For each β in a small grid around the 4D U(1) bulk transition (β ≈ 1.01):

1. **Stage 1 — plain gauge HMC** (`apps/u1_hmc`). Per-trajectory Wilson
   action `S` goes into a histogram — an importance-sampled `ρ(S)`
   estimate in the typical-S region only.
2. **Range selection.** Reads the HMC `S` chain, takes its central
   percentile range and pads it; uses that to set the LLR window
   `[E_min, E_max]`. δ is auto-chosen to give ~`TARGET_N_REP` replicas.
3. **Stage 2 — gauge LLR** (`apps/u1_llr`). NR warm-up + restarted RM +
   alternating even/odd nearest-neighbour replica exchange. Outputs the
   per-replica LLR slopes `a_n`.
4. **Comparison** (`analyze.py`). Reconstructs `ln ρ(S)` by trapezoidal
   integration of `a(E)` along the replica grid, aligns to the HMC peak,
   and overlays both. Adds a panel showing `a_n` convergence across the
   NR + RM iterations.

## Literature target

The 4D compact U(1) ρ(S) at the (weakly) first-order bulk transition is
the original LLR showcase result — [Langfeld, Lucini, Rago, *PRL* 109,
111601 (2012)](https://arxiv.org/abs/1204.3243) showed the characteristic
double-peak structure that plain HMC sees only one side of, while LLR
resolves both peaks plus the saddle between them.

The shipped defaults use `L = 4`, which is small for finite-size
purposes — at `L = 4` the bulk transition is smeared into a crossover —
but the LLR ρ(S) curves still show clear shape evolution as β crosses
the transition. To see a sharper double peak: raise `L` (cost ~ L⁴, so
keep an eye on wall time) and use `BETAS="1.00 1.01 1.02"` to bracket
the transition more tightly.

## Run it

```bash
# 1. build (once)
cmake --build --preset macos-appleclang

# 2. run both stages + compare
bash examples/05_u1_llr/run.sh
python3 examples/05_u1_llr/analyze.py
```

Outputs in `examples/05_u1_llr/results/`:
* `hmc_beta<b>.h5`   — full HMC output (per-trajectory S, plaquette, etc.)
* `range_beta<b>.txt`— `E_min E_max delta` chosen for the LLR run
* `llr_beta<b>.h5`   — LLR slopes per replica + exchange stats

And `rho_hmc_vs_llr.png` next to the script: per-β subplot, HMC histogram
markers + aligned LLR `ln ρ(S)` line on top; `a_n` convergence beneath.

## Knobs (env vars)

| variable          | default            | meaning                                       |
| ----------------- | ------------------ | --------------------------------------------- |
| `RETICOLO_PRESET` | `macos-appleclang` | which build preset to invoke                  |
| `JOBS`            | `nproc`            | parallel β values at once                     |
| `NDIM`            | `4`                | spatial dimensions                            |
| `L`               | `4`                | linear lattice extent                         |
| `BETAS`           | `0.95 1.00 1.05`   | space-separated β list                        |
| `N_THERM`         | `2000`             | HMC thermalisation trajectories               |
| `N_PROD`          | `15000`            | HMC production trajectories                   |
| `TARGET_N_REP`    | `30`               | replica count used by the auto-δ heuristic    |
| `DELTA`           | `0` (auto)         | Gaussian half-width = replica spacing; 0 ⇒ auto |
| `RANGE_PAD`       | `0.30`             | fractional padding added on each side         |
| `RANGE_LO_PCT`    | `0.5`              | lower percentile of HMC S used as range low   |
| `RANGE_HI_PCT`    | `99.5`             | upper percentile of HMC S used as range high  |
| `N_NR`, `N_RM`    | `6`, `40`          | NR warm-up / RM sweep counts                  |
| `N_THERM_NR`, `N_MEAS_NR` | `200`, `1000` | HMC traj counts inside each NR iter           |
| `N_THERM_RM`, `N_MEAS_RM` | `100`, `500`  | HMC traj counts inside each RM sweep          |
| `TAU`, `N_MD`     | `1.0`, `20`        | HMC trajectory length / MD steps              |
| `SEED`            | `20260518`         | RNG seed                                      |

The defaults are sized for a quick demo run (a few minutes on a laptop).
For sharper tails / a publishable replication of the LLR double peak,
raise `L` and `N_RM` and lower `DELTA`.

## Paper-replication run

`run_paper.sh` targets the bulk first-order transition shown in
[Langfeld, Lucini, Pellegrini, Rago, *EPJC* 76:306 (2016)](https://arxiv.org/abs/1509.08391)
— `L=6`, `ndim=4`, three β values straddling β_c ≈ 1.0106(18). Heavier
sample counts than the quick demo so the LLR can resolve the double-peak /
shoulder structure of `ρ(E_p)` at the transition. Approx wall time on 3
cores at the shipped defaults: ~1.5 h.

```bash
bash examples/05_u1_llr/run_paper.sh
python3 examples/05_u1_llr/analyze.py
```

For a cleaner double peak — what's actually shown in the paper figures — raise
`L` to 8 (factor ~5 in wall time) and lengthen `N_RM` correspondingly.

This won't reproduce the paper's *Potts* results (need a new discrete-spin
action), nor its complex-action work — both are out of scope for the
current tree.
