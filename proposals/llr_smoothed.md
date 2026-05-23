# LLR smoothed: cross-replica smoothing in the RM phase

## Motivation

Standard LLR runs each replica's Robbins–Monro recursion independently and
combines the resulting `a_n` only at post-processing (reconstruction of
`ln ρ(E)`). The √k per-replica rate is the local Cramér–Rao bound for the
single-root problem — but the `a_n` are samples of a *function* `a*(E)`
which is smooth away from phase transitions. Pooling information across
replicas during the RM phase can collapse the transient and reduce the
finite-time error, with no change to the asymptotic fixed point.

This proposal sketches a second LLR driver, `reticolo::llr::smoothed::run`,
that performs cross-replica shrinkage at every RM step. The vanilla
`llr::run` is unchanged.

## Provably-unbiased design

For each replica `n` at RM step `k`, the existing recursion is

    a_{n,k+1} = a_{n,k} + c_k · <dE>_{n,k},     c_k = 1 / (δ² (k+1))

The smoothed variant adds a shrinkage term toward a smoothed estimate
`â_k(E)` fit across replicas:

    a_{n,k+1} = a_{n,k} + c_k · <dE>_{n,k} − λ_k · (a_{n,k} − â_k(E_n))

with `λ_k = λ_0 / (k+1)^(1+ε)`, `ε > 0`. Two properties matter:

1. **`Σ λ_k < ∞`.** The cumulative shrinkage perturbation is finite, so
   Kushner–Yin perturbed-RM gives a.s. convergence of `a_{n,k}` to the
   per-replica zero `a*(E_n)` of `E[<dE>]`, *regardless of whether â is a
   good fit*. The fixed point is unchanged from vanilla LLR. This is the
   key safety property: a misspecified smoother (or a phase transition the
   smoother fails to detect) cannot bias the final answer.

2. **Transient acceleration.** Early in the run, `λ_k = O(1)` so the
   shrinkage dominates and the replica's `a_n` is pulled toward a much
   lower-variance estimate built from O(K) neighbouring replicas. The
   asymptotic √k rate is preserved but the constant shrinks roughly like
   `1/√K` over the smoothing window.

The same argument applies to the NR warm-up, but we keep NR unchanged in
the first cut — the NR step is already O(1) iterations and the benefit
is small.

### Choice of λ schedule

Default: `λ_k = λ_0 / (k+1)^2` with `λ_0 = 1`. Clearly summable, kicks
in hard during the first few RM steps, negligible by `k ~ 10`. The
schedule is a CLI knob.

## The smoother

First cut: **local quadratic fit in E** with fixed bandwidth K (number
of neighbours each side). For each replica `n`, fit a parabola through
`(E_m, a_{m,k})` for `m ∈ [n−K, n+K]` and evaluate at `E_n`.

A linear fit would force `a(E)` locally affine, i.e. `ln ρ(E)` locally
quadratic — pure Gaussian. That's exactly the assumption that breaks
near a transition, where the curvature of `a(E)` (≡ third derivative of
`ln ρ`, related to the specific-heat derivative) is the interesting
signal. Quadratic is the minimum degree that lets the local DoS shape
deviate from Gaussian.

Defaults: degree `p = 2`, bandwidth `K = 4` (9-point fit per replica),
both CLI knobs. Boundary replicas use the same K but the asymmetric
window — the local-polynomial fit handles that gracefully. We require
`2K + 1 ≥ p + 2` (one extra DoF beyond a determined fit) so K=2 is the
floor for quadratic; K=4 gives 6 DoF over 3 parameters, enough that
the per-replica fit isn't dominated by the central point itself.

The fit is a small weighted least-squares solve per replica per RM
step — O(n_rep · K · p²), negligible next to the MC sweeps. Use the
QR / normal-equations solver from the existing math headers (or just
roll it inline; 3×3 symmetric).

Two principled upgrades exist for later:

- **Lepski-adaptive K.** Per-replica, pick the largest K such that the
  fitted â_n at smaller K's agrees within a confidence band built from
  `<dE>` variance. Automatically collapses to K=`(p+1)/2` (degenerate
  fit) near a phase transition. Reiß–Wahl 2014 / Spokoiny–Vial 2011
  give the selection rule.
- **GP regression with Matérn-5/2 kernel** (twice-differentiable
  sample paths, matches a quadratic local model). Pricier but gives
  explicit posterior variance, which lets us report uncertainty bands
  on `â(E)` and drive active replica allocation later. Probably not
  worth it until the local-quadratic version is benchmarked.

## File layout

```
include/reticolo/llr/
  smoothed_driver.hpp       run() + local-polynomial fit in a detail::
                            namespace. Single file mirroring llr/driver.hpp.

apps/
  u1_llr_smoothed.cpp       copy of u1_llr.cpp, swap llr::run for
                            llr::smoothed::run, extra CLI: --smooth_K --smooth_lambda0

examples/
  09_u1_llr_smoothed/
    run_critical_compare.sh   sweep both u1_llr and u1_llr_smoothed at the
                              same β, same seed, same δ
    analyze.py                overlay <dE>/δ vs k for the two; compare
                              reconstructed ρ(E)
```

## Recursion sketch (driver)

```cpp
// Same NR phase as llr::run, then:
for (int s = 0; s < spec.n_rm; ++s) {
    // 1. Per-replica RM step in parallel (unchanged from llr::run)
    #pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t n = 0; n < n_rep_u; ++n) {
        auto _    = log::scope(reps[n]->id());
        auto& r   = *reps[n];
        r.thermalize(spec.n_therm_rm, log::Mode::silent);
        de_buf[n] = r.sample(spec.n_meas_rm, log::Mode::silent);
        a_rm[n]   = rm_update(r.a(), de_buf[n], spec.delta, s);
    }

    // 2. Global smoothing fit (serial, cheap: O(n_rep · K · p²))
    smoothed::local_poly_fit(e_n_vec, a_rm,
                           /*K=*/spec.smooth_K, /*deg=*/spec.smooth_degree,
                           a_hat);

    // 3. Shrinkage (serial: O(n_rep))
    double const lam = spec.smooth_lambda0 / std::pow(s + 1.0, spec.smooth_lambda_exp);
    for (std::size_t n = 0; n < n_rep_u; ++n) {
        a_buf[n] = (1.0 - lam) * a_rm[n] + lam * a_hat[n];
        reps[n]->set_a(a_buf[n]);
    }

    // 4. Drain HDF5 buffers + exchange (unchanged)
}
```

The smoother and shrinkage are serial — they're O(n_rep · K) and O(n_rep)
respectively, dwarfed by the MC sweeps.

## Output schema additions

Same `/replica_NNN/a` and `/replica_NNN/dE` as `llr::run`, plus:

- `/replica_NNN/a_pre_shrink` — `a_rm[n]` before the shrinkage step
- `/replica_NNN/a_hat`        — `â_k(E_n)` from the smoother
- `/cfg@smooth_K`, `/cfg@smooth_degree`, `/cfg@smooth_lambda0`, `/cfg@smooth_lambda_exp`

The two extra series make the bias diagnostic trivial: at convergence
`a` and `a_pre_shrink` must agree to the noise floor; `a_hat - a` shows
the residual smoothness bias.

## Validation on U(1) at critical β

Test on a 4D compact U(1) Wilson action near the bulk transition
(β ≈ 1.01 on L=4–8, where the existing `05_u1_llr` example already
shows non-trivial double-peak structure). The HMC is not trivial in
that region and RM convergence is slow, so any acceleration shows up
clearly.

Falsification tests:

1. **Bias check.** Same `δ`, same seed, run both `u1_llr` and
   `u1_llr_smoothed` to convergence (long n_rm). Reconstructed ρ(E) must
   agree to within the per-replica statistical uncertainty. Any
   systematic shift = bias bug.

2. **Acceleration check.** At fixed `n_rm`, the smoothed variant should
   have smaller residual `|<dE>/δ|` across all replicas. Plot
   `max_n |<dE>/δ|` vs `k` for both. Expect the smoothed curve to drop
   below the vanilla curve from the first few steps and stay below.

3. **Transition robustness.** Crank `δ` down or move into a sharper
   feature so `a*(E)` has a kink within the replica range. The smoothed
   variant should still converge (eventually) to the same ρ(E) as
   vanilla — the summable `λ_k` guarantees this. If we see persistent
   disagreement, the schedule needs adjusting.

## What this proposal does *not* do

- No active replica placement (no GP, no information-gain weighting).
- No Lepski-adaptive bandwidth — fixed K.
- No SPRB / probabilistic-bisection per-replica updaters.
- No change to `llr::run`, NR phase, exchange, or any existing app.

These are deliberate. We want the smallest possible incremental change
that lets us measure (a) whether the bias story holds in practice and
(b) whether the transient acceleration is worth the complexity.

## References

- Mokkadem & Pelletier, *Ann. Stat.* 2007 — kernel-smoothed SA, the
  framework that justifies cross-replica pooling.
- Doan, *Automatica* 2024 (arXiv:2010.15088) — distributed local SA
  with graph-Laplacian coupling, structural match to the smoothed LLR
  setup.
- Wai et al. 2017 (arXiv:1712.08817) — multi-task SA with Laplacian
  regularisation; gives the explicit bias formula for non-vanishing
  shrinkage.
- Langfeld, Lucini, Pellegrini, Rago, *EPJ C* 76 (2016) 306
  (arXiv:1509.08391) — original LLR convergence proof; smoothed variant
  inherits its assumptions (replica ergodicity, well-posed window).
- Reiß & Wahl, *Ann. Stat.* 2014 (arXiv:1305.6430) — Lepski adaptive
  bandwidth, the future upgrade for phase-transition handling.
