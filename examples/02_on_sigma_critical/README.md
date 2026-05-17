# 02 — O(3) sigma model critical point

Locates the critical inverse temperature β_c of the 3D O(3) sigma model by
sweeping β at several lattice sizes and reading off the crossing of the
Binder cumulants. β_c is known analytically/numerically to sit near
**0.693**; the example reproduces that crossing.

## What this proves

- The library can drive a real finite-size-scaling measurement on a vector-
  valued field.
- `obs::vector_magnetization_sq<N>` plus per-configuration squaring on the
  Python side is enough to assemble the Binder cumulant — no extra observer
  needed on the C++ side.

## Run it

```bash
# 1. build (once)
cmake --build --preset macos-appleclang

# 2. sweep + analyse
bash examples/02_on_sigma_critical/run.sh
python3 examples/02_on_sigma_critical/analyze.py
```

Outputs land in `examples/02_on_sigma_critical/results/` (one HDF5 per
`(L, β)`) and `binder.png` next to the script.

## Knobs

| variable          | default              | meaning                          |
| ----------------- | -------------------- | -------------------------------- |
| `RETICOLO_PRESET` | `macos-appleclang`   | which build preset to invoke     |
| `SIZES`           | `4 6 8`              | space-separated list of `L`s     |
| `NDIM`            | `3`                  | spatial dimensions               |
| `N_THERM`         | `500`                | Metropolis thermalisation sweeps |
| `N_PROD`          | `3000`               | Metropolis production sweeps     |
| `SEED`            | `20260517`           | RNG seed                         |

The β grid is hard-coded inside `run.sh`; widen or refine there.

## Reading the plot

Each curve `U(β; L)` is monotonically decreasing through the transition.
Curves for different `L` cross within a narrow β-window — the crossing
locates β_c up to finite-size corrections. For the O(3) universality class
all curves should converge to the universal value `U*` ≈ 0.628 at that
point.

## Extending

- Switch `N = 3` to `N = 2` in `apps/on_sigma_metropolis.cpp` and re-run to
  recover the XY universality class (then the Wolff updater `apps/xy_wolff`
  becomes the right tool for the same study with much smaller
  autocorrelation times).
- Add a Wolff updater equivalent for O(N) by giving `OnSigma` a sibling app
  that drives `alg::Wolff` instead of `alg::Metropolis`.
