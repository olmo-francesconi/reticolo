# 02 — 3D O(3) Heisenberg critical point via Wolff cluster

Locates the critical inverse temperature β_c of the 3D O(3) sigma model and
extracts the correlation-length exponent 1/ν jointly from a Bhattacharjee-Seno
collapse of the Binder cumulant. Uses **`alg::Wolff`** single-cluster updates
(`apps/on_sigma_wolff.cpp`) — cluster moves crush autocorrelation near β_c
where Metropolis would need orders of magnitude more sweeps for the same
error bars.

The literature target (Campostrini et al. 2002, high-precision improved-action
MC) is **β_c ≈ 0.693002**, **ν = 0.7112(5)** ⇒ **1/ν ≈ 1.406**.

## What this proves

- `obs::vector_magnetization_sq<N>` plus per-configuration squaring on the
  Python side is enough to assemble the Binder cumulant — no extra observer
  on the C++ side.
- The Wolff updater drives the system at criticality without thermalisation
  pathologies; the per-point errors are jackknife with proper binning.
- Joint extraction of (β_c, 1/ν) via collapse minimisation lands within ~0.1%
  on β_c and a few percent on 1/ν at the shipped defaults.

## Building & running

Standalone build:

```sh
cd examples/02_on_sigma_critical
cmake -S . -B build && cmake --build build
```

Or let `run.sh` do it automatically:

```sh
bash examples/02_on_sigma_critical/run.sh
python3 examples/02_on_sigma_critical/analyze.py
```

Override the preset with `RETICOLO_PRESET=macos-llvm ./run.sh`.

## Run it

Results land in `examples/02_on_sigma_critical/results/` (one HDF5 per (L, β))
and `binder.png` next to the script.

## What you get

- **Left panel**: `U(β)` for each `L`, every curve a different colour with
  X markers and error bars, fitted β_c and literature β_c drawn as vertical
  reference lines.
- **Right panel**: collapsed `U` vs `(β − β_c)·L^{1/ν}` using the fitted 1/ν
  — every L's data falls on a single curve.
- **Stdout**: fitted β_c and 1/ν with the literature reference values.

Typical result at the shipped defaults (5 sizes L = 6–20, 90 sweeps, ~1 min
wall on 8 cores): `β_c` within 0.2% of literature, `1/ν` within 3%.

## Knobs

| variable          | default              | meaning                          |
| ----------------- | -------------------- | -------------------------------- |
| `RETICOLO_PRESET` | `macos-appleclang`   | which build preset to invoke     |
| `JOBS`            | `nproc`              | parallel runs at once            |
| `SIZES`           | `6 8 12 16 20`       | space-separated list of `L`      |
| `NDIM`            | `3`                  | spatial dimensions               |
| `N_CLUSTER`       | `5`                  | Wolff updates per measurement    |
| `N_THERM`         | `2000`               | thermalisation measurements      |
| `N_PROD`          | `30000`              | production measurements          |
| `SEED`            | `20260517`           | RNG seed                         |

The β grid is hard-coded inside `run.sh`; widen or refine there.

## Reading the plot

Each curve `U(β; L)` rises monotonically through the transition; curves for
different `L` cross within a narrow β-window — that crossing locates β_c up
to finite-size corrections. The universal value `U*` ≈ 0.625 at the crossing
matches the O(3) crossing of the Binder cumulant in the literature.

## Extending

- Switch `N = 3` to `N = 2` in `apps/on_sigma_wolff.cpp` to recover the 3D XY
  universality class (β_c ≈ 0.4542, ν ≈ 0.6717 — Campostrini-Pelissetto 2001),
  then update the literature constants in `analyze.py`.
- Pair with `apps/on_sigma_metropolis.cpp` at the same β to demonstrate the
  autocorrelation-time gap between Metropolis and Wolff near criticality.
