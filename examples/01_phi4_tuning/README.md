# 01 — φ⁴ FSS in the 2D Ising universality class

End-to-end finite-size-scaling study of the **2D** φ⁴ scalar field across the
broken-symmetry transition. Sweeps κ at fixed λ for several `L`, computes the
connected susceptibility per lattice with block-jackknife errors, and extracts
`(κ_c, γ/ν, 1/ν)` from a Bhattacharjee-Seno collapse fit.

The target is the exact 2D Ising universality class — Onsager 1944:
`ν = 1`, `η = 1/4`, `γ/ν = 7/4`, `1/ν = 1`.

## What this proves

- The whole `apps/phi4_hmc.cpp` → HDF5 → Python loop works from a fresh clone
  with two commands.
- The library reproduces a textbook FSS collapse on three independent
  observables (`κ_c`, `γ/ν`, `1/ν`) extracted jointly by minimising a
  Bhattacharjee-Seno residual on (κ, χ) data from multiple `L`.
- All errors are jackknife (autocorrelation-aware), every plot point has its
  own error bar, the master curve is *not* drawn (just the raw rescaled data
  collapsing onto itself).

## Run it

```bash
# 1. build (once)
cmake --build --preset macos-appleclang

# 2. sweep (parallel across cores) + analyse
bash examples/01_phi4_tuning/run.sh
python3 examples/01_phi4_tuning/analyze.py
```

Results land in `examples/01_phi4_tuning/results/` (one HDF5 per (L, κ)) and
`susceptibility.png` next to the script.

## What you get

- **Left panel**: raw `χ(κ)` for each `L`, error bars, peaks growing with `L`,
  the fitted κ_c drawn as a dashed vertical line.
- **Right panel**: rescaled `χ·L^{−γ/ν}` vs `(κ − κ_c)·L^{1/ν}` using the
  *fitted* exponents — every L's data should fall on one master curve.
- **Stdout**: `κ_c`, `γ/ν`, `1/ν` with their literature reference values.

Typical result at the shipped defaults (4 sizes L = 16–48, 68 sweeps,
~1 min wall on 8 cores): `γ/ν` ≈ 1.78 (exact 1.75, ~2% off), `1/ν` ≈ 1.11
(exact 1.0, ~11% off). The residual on `1/ν` is the leading
correction-to-scaling (`L^{−ω}` with `ω ≈ 0.8` in 2D Ising) — to close that
gap you'd either run at substantially larger L or fit a correction term
explicitly.

## Knobs

| variable          | default              | meaning                          |
| ----------------- | -------------------- | -------------------------------- |
| `RETICOLO_PRESET` | `macos-appleclang`   | which build preset to invoke     |
| `JOBS`            | `nproc`              | parallel runs at once            |
| `SIZES`           | `16 24 32 48`        | space-separated list of `L`      |
| `NDIM`            | `2`                  | spatial dimensions               |
| `LAMBDA`          | `1.145`              | quartic coupling                 |
| `N_THERM`         | `3000`               | HMC thermalisation trajectories  |
| `N_PROD`          | `50000`              | HMC production trajectories      |
| `SEED`            | `20260517`           | RNG seed                         |

The κ grid is hard-coded inside `run.sh`; widen or refine there.

## Extending

- `NDIM=3 SIZES="12 16 24" LAMBDA=1.145 bash run.sh` switches to 3D Ising
  (literature: γ/ν ≈ 1.96, 1/ν ≈ 1.59 from conformal bootstrap). Update the
  literature constants in `analyze.py` accordingly.
- Use `phi6_hmc` instead and add a `--g6` flag to study the tricritical region.
