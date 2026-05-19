# 07 — SU(2) gauge density of states with LLR + replica exchange

End-to-end demo of the matrix-link gauge stack (`MatrixLinkLattice<SU2,
double>` + `Wilson<SU2>` + `Replica<...>`) on 4D SU(2) Wilson gauge
theory. Mirrors example 05 (U(1)) but with the generic gauge group plug.

## What this example does

For each β in a small grid in the intermediate / weak-coupling regime:

1. **Stage 1 — plain SU(2) HMC** (`apps/su2_hmc`). Per-trajectory Wilson
   action `S` goes into a histogram — an importance-sampled `ρ(S)`
   estimate in the typical-S region only.
2. **Range selection.** Reads the HMC `S` chain, takes its central
   percentile range and pads it; uses that to set the LLR window
   `[E_min, E_max]`. δ is auto-chosen to give ~`TARGET_N_REP` replicas.
3. **Stage 2 — SU(2) LLR** (`apps/su2_llr`). NR warm-up + restarted RM +
   alternating even/odd nearest-neighbour replica exchange. Outputs the
   per-replica LLR slopes `a_n`.
4. **Comparison** (`analyze.py`). Reconstructs `ln ρ(S)` by trapezoidal
   integration of `a(E)` along the replica grid, aligns to the HMC peak,
   and overlays both. Adds a panel showing `a_n` convergence across the
   NR + RM iterations.

## Why no first-order transition

Unlike compact U(1) in 4D (example 05), SU(2) Wilson in 4D has no sharp
bulk transition — it's a smooth crossover from strong to weak coupling.
The demonstration value is therefore not the LLR resolving a metastable
double-peak, but rather LLR's reach into the suppressed tails of ρ(S),
many orders of magnitude below where the HMC histogram has any signal.
The three β panels shift the peak of ρ progressively leftward (toward
small S = ordered configurations) as β grows, with the LLR
piecewise-exponential reconstruction extending smoothly into both tails.

## Convention

This example uses the standard Wilson convention `S_W = (β/N)·Σ_p (N − Re
Tr U_p)` (so `S_W ≥ 0`, weight `exp(−S)`, `H = K + S`). Mode-A LLR with
`a_init = 0` corresponds to the natural Boltzmann ensemble being the
neutral starting point. The reconstructed quantity is `ln ρ_natural(S) =
∫ a(E) dE`, matching the formula in examples 04 (φ⁴) and 05 (U(1)).

## How to run

```sh
cmake --build --preset macos-appleclang -j
./run.sh         # ~5 min on 8 cores at L=4 4D defaults
python3 analyze.py
open rho_hmc_vs_llr.pdf
```

Tunable knobs (env vars):

| var               | default | purpose                                 |
|-------------------|---------|-----------------------------------------|
| `BETAS`           | "2.0 2.2 2.4" | comma-/space-separated β grid (see note below) |
| `L` / `NDIM`      | 4 / 4   | lattice extent and dimensions           |
| `N_THERM` / `N_PROD` | 1000 / 8000 | HMC trajectories for histogram   |
| `N_NR` / `N_RM`   | 6 / 30  | LLR Newton-Raphson + Robbins-Monro iters|
| `TARGET_N_REP`    | 30      | auto-set δ to give ~this many replicas  |
| `JOBS`            | nproc   | total CPU budget                        |
| `OMP_THREADS`     | 4       | threads each LLR job uses internally    |

## β range — LLR stability boundary

The default β grid stops at 2.4 by design. At β ≳ 2.5 on L = 4, the
natural HMC S-distribution narrows enough that the auto-tuned δ (chosen
from HMC percentiles to give `TARGET_N_REP` ≈ 30 replicas) becomes small
enough that the LLR Gaussian-penalty force scale `(S − E_n)/δ²` makes
the inner-replica HMC integrator unstable: `a` runs to large values
without settling, and ρ(S) is junk. To push to higher β on L = 4, either
drop `TARGET_N_REP` (gives wider windows / larger δ) or pin `DELTA`
explicitly; the auto-δ is conservative for the intermediate-coupling
regime, not for deep weak coupling. Larger volumes don't have this
problem because the HMC distribution widens roughly like √V.

## Output

`results/hmc_beta{...}.h5`, `results/llr_beta{...}.h5` (raw HDF5 with the
schema from §3 of the apps), and `rho_hmc_vs_llr.pdf` (the four-row
per-β panel from `analyze.py`).
