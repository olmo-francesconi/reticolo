# 09 — Smoothed LLR vs vanilla LLR (synthetic benchmark)

**Python-only.** Unlike the other numbered examples this carries no C++ driver
or `CMakeLists.txt` — it does **not** link `reticolo`. It studies the LLR
*algorithm* (the smoothing layer behind `llr/smoothed_driver.hpp`) in isolation,
with no Monte Carlo and no HDF5.

## What it does

Picks a ground-truth slope profile `a*(E)`, generates fake `⟨dE⟩` samples under
the linear-response model

    ⟨dE⟩(a, E_n) = δ²·(a*(E_n) − a) + ε,   ε ~ N(0, σ²)

and drives both vanilla and smoothed LLR on the **same** noise sequence per seed,
so any difference is purely the smoothing layer. Three profiles (linear /
sigmoid / double-peak); per profile one PDF with RMSE-vs-iteration (±1σ over
seeds), true-vs-reconstructed `a*(E)`, per-replica final-step bias, and example
replica trajectories.

## Run

```sh
python3 benchmark.py
```
