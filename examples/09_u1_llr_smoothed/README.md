# 09 — smoothed LLR vs vanilla LLR

Direct side-by-side comparison of `u1_llr` and `u1_llr_smoothed` at the
4D compact U(1) bulk transition (β = 1.002, L = 6, same setup as
example 05). Identical RNG seed and physics; only the driver
differs.

## Background

Vanilla LLR runs an independent Robbins–Monro recursion per replica.
Smoothed LLR fits a local quadratic across the per-replica `a_n` at
every RM step and shrinks each replica's iterate toward the fitted
value:

    a_rm[n]  = a[n] + ⟨dE⟩[n] / (δ² (s+1))
    â        = local_poly_fit({(E_m, a_rm[m])}, K=4, deg=2)
    λ_s      = λ₀ / (s+1)^α                 (α=2 → summable)
    a[n] ← (1 − λ_s) · a_rm[n] + λ_s · â[n]

With α > 1 the shrinkage perturbation is summable, so by Kushner–Yin
the fixed point of the iteration is unchanged from vanilla RM. The
smoother only collapses the transient — it cannot introduce
asymptotic bias even if `a*(E)` has a feature the quadratic fit
misses.

See `proposals/llr_smoothed.md` for the design + references.

## What this example shows

- **Convergence (left panel of the output PDF):** `max_n |⟨dE⟩_n|/δ`
  vs RM iteration. The smoothed curve should drop faster early on.
- **Reconstruction agreement (right panel):** `ln ρ(S)` from
  tail-median of the RM history must overlay between the two runs to
  within the per-replica statistical noise — this is the empirical
  unbiasedness check.

## Run

```sh
cmake --build --preset macos-appleclang   # builds u1_llr and u1_llr_smoothed
./run.sh                                   # ~6 min on 8 cores
```

Outputs:

- `results/llr_vanilla.h5`, `results/llr_smoothed.h5` — raw run data
- `compare_vanilla_vs_smoothed.pdf` — convergence + reconstruction overlay
- `doublepeak_vanilla.pdf`, `doublepeak_smoothed.pdf` — full 2x2 panel
  (ln ρ, ρ, a_n convergence, a_n vs E_n), one per algorithm, same
  layout as example 05
- `smoothed_amplitudes.pdf` — per-step breakdown of the smoothed update:
  magnitude of the RM contribution `|a_rm − a_old|` vs the smoothing
  contribution `|λ · (â − a_rm)|`, plus the λ schedule. Shows
  visually when the smoother dominates and when it fades out.
- `rho_evolution_side_by_side.gif` — frame-by-frame DoS evolution,
  vanilla on the left, smoothed on the right, sharing axes for direct
  visual comparison

## Knobs worth sweeping

- `smooth_K` (neighbours each side of the local fit). Default 4. Larger
  → more smoothing, smaller transient if `a(E)` is genuinely smooth.
- `smooth_degree`. Default 2. Linear (1) forces local Gaussian DoS and
  will visibly bias near a transition. Cubic (3) needs `K ≥ 3` for
  determinacy.
- `smooth_lambda_exp`. Default 2. Set to 1 (non-summable) to let the
  smoother drive the asymptotics — buys faster rate at the cost of
  a non-vanishing bias proportional to `||smoothness violation||`.
  Don't ship this; it's a knob for studying the bias/rate tradeoff.
