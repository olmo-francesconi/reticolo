# 01 — φ⁴ kappa tuning

End-to-end study of the 4D φ⁴ scalar field across the broken-symmetry
transition. Sweeps the hopping parameter κ at fixed quartic coupling λ and
extracts the magnetic susceptibility.

## What this proves

- The whole `apps/phi4_hmc.cpp` → HDF5 → Python loop works from a fresh
  clone with two commands.
- The library reproduces the standard susceptibility peak signalling the
  Z₂-breaking transition: `χ = V(⟨m²⟩ − ⟨|m|⟩²)` rises sharply around the
  critical κ, with finite-size rounding controlled by `L`.

## Run it

```bash
# 1. build (once)
cmake --build --preset macos-appleclang   # or your preset of choice

# 2. sweep + analyse
bash examples/01_phi4_tuning/run.sh
python3 examples/01_phi4_tuning/analyze.py
```

Outputs land in `examples/01_phi4_tuning/results/` (one HDF5 per κ) and
`susceptibility.png` next to the script.

## Knobs

Environment variables override the defaults:

| variable          | default              | meaning                          |
| ----------------- | -------------------- | -------------------------------- |
| `RETICOLO_PRESET` | `macos-appleclang`   | which build preset to invoke     |
| `L`               | `6`                  | linear lattice extent (4D ⇒ V=L⁴) |
| `LAMBDA`          | `0.02`               | quartic coupling                 |
| `N_THERM`         | `200`                | thermalisation trajectories      |
| `N_PROD`          | `2000`               | production trajectories          |
| `SEED`            | `20260517`           | RNG seed                         |

The κ grid is hard-coded inside `run.sh`; edit there to widen or refine.

## Extending

- Replace `phi4_hmc` with `phi6_hmc` and add a `--g6` flag to study the
  tricritical region.
- Add a second loop over `L` and overlay susceptibility curves to read off
  finite-size scaling.
