# LLR cache wiring — base-action raw-value cache + HMC reject rollback

Status: draft / pre-implementation, 2026-05-20.

Follow-up to `ddee54d` (`refactor(hmc): split hamiltonian_ into kinetic_ + s_full, cache last_s_full`). That commit landed the HMC-level cache (`Hmc::last_s_full_`) but neither of its intended consumers is wired. This proposal wires the one that matters in the short term (`Replica::sample()` skipping its post-trajectory constraint sweep) under the principle: **base actions cache raw scalars; the LLR layer composes weights on top.**

## Goal

Eliminate the redundant `windowed_.constraint_value(phi_)` full-lattice sweep at `replica.hpp:84`, saving ~1 sweep per trajectory in `Replica::sample()`. Keep `Hmc::last_s_full_` (windowed value, ready for Replica Exchange) unchanged.

## Separation of concerns

| Layer | Caches | Read by |
|---|---|---|
| Base action (`Wilson<G>`, `BoseGas`) | `last_s_full_`; `BoseGas` also caches `last_s_imag_` | `Replica::sample()` |
| `Hmc` | `last_s_full_` = `action_.s_full(field_)` (== `S_LLR` when wrapped in `WindowedAction`) | future `Exchange` |
| `WindowedAction` | nothing — pure tilt layer | — |

## Strategy: concept-gated checkpoint/restore

Base action exposes:

```cpp
[[nodiscard]] scalar_t last_s_full() const noexcept;          // NaN before first call
void restore_last_s_full(scalar_t v) const noexcept;          // setter for HMC rollback
```

(Plus `_imag` variants on `BoseGas`.) `s_full()` / `s_imag()` cache into `mutable scalar_t last_s_full_` before returning.

`Hmc` defines a concept and gates snapshot/restore:

```cpp
template <class A>
concept HasCacheRollback = requires(A& a, scalar_of_t<typename A::value_type> v) {
    { a.last_s_full() } noexcept;
    a.restore_last_s_full(v);
};
```

`trajectory()` snapshots after h0, restores on reject:

```cpp
double const s0 = static_cast<double>(action_.s_full(field_));   // caches q0 in base
auto const cache_snap = capture_action_cache_(action_);          // empty for non-cache actions
double const h0 = kin0 + s0;

// ... MD ...

double const s1 = static_cast<double>(action_.s_full(field_));   // overwrites cache with q1
// ... accept/reject ...
if (!accepted) {
    // field rollback ...
    restore_action_cache_(action_, cache_snap);                  // q1 -> q0
    last_s_full_ = s0;
} else {
    last_s_full_ = s1;
}
```

`capture_action_cache_` / `restore_action_cache_` are private constexpr-dispatched helpers on `Hmc`, returning an empty struct for non-cache-bearing actions and the captured scalar(s) for those that satisfy the concept. `WindowedAction` does **not** satisfy the concept — HMC reaches the base via either explicit forwarding or by having `WindowedAction` itself satisfy the concept as a pass-through. Pick at implementation time; the pass-through is one extra method on `WindowedAction` and keeps HMC unaware of `WindowedAction` specifically.

## File-by-file changes

**`include/reticolo/action/wilson.hpp`** (~10 lines)
- Add `mutable scalar_t last_s_full_ = quiet_NaN`.
- In `s_full(U)`: assign `last_s_full_` before returning.
- Add `last_s_full()` getter, `restore_last_s_full(v)` setter.

**`include/reticolo/action/bose_gas.hpp`** (~20 lines)
- Same as Wilson for `s_full`.
- Plus `mutable scalar_t last_s_imag_`, cache in `s_imag(l)`, `last_s_imag()` + `restore_last_s_imag(v)`.

**`include/reticolo/llr/windowed_action.hpp`** (~6 lines, optional pass-through)
- `last_s_full() noexcept { return base.last_s_full(); }`
- `restore_last_s_full(v) noexcept { base.restore_last_s_full(v); }`
- Same for `_imag` when `k_complex`.
- Or omit and have `Hmc` look at `windowed_.base` directly — uglier, couples HMC to WindowedAction. Pass-through preferred.

**`include/reticolo/algorithm/hmc.hpp`** (~20 lines)
- Define `HasCacheRollback` concept (above).
- Private helpers `capture_action_cache_()` / `restore_action_cache_()` returning a `CacheSnap` struct (empty when concept fails; `scalar` or `(scalar, scalar)` when it holds).
- Wire snapshot after the h0 `s_full` call, restore in the reject branch.
- Keep `last_s_full_` member exactly as today.

**`include/reticolo/llr/replica.hpp`** (~3 lines)
- `sample()` line 84: replace `windowed_.constraint_value(phi_)` with:
  - mode A: `windowed_.base.last_s_full()` (or the WindowedAction pass-through)
  - mode B: `windowed_.base.last_s_imag()`
- Helper to pick the right one at compile time (or just `windowed_.last_constraint_value()` that dispatches).

**All real-scalar actions** get the same `s_full` cache pattern: `compact_u1.hpp`, `on_sigma.hpp`, `phi4.hpp`, `phi6.hpp`, `sine_gordon.hpp`, `xy.hpp`. Uniform across the action tree — any future LLR or measurement consumer reads the cache without per-action plumbing. The cache slot is a public `mutable T last_s_full_` so the structs stay aggregate-initializable (`Phi4{.kappa = ..., .lambda = ...}` in the apps).

## Non-goals (deferred)

- **Replica Exchange driver** — `Hmc::last_s_full_` and `Replica::last_windowed_action()` are already wired for it; ship when Exchange itself is added.
- **Per-MD-step `s_full` elimination inside `compute_force_and_kick`** — needs incremental updates or fusion at the gauge-group layer, called out in the original commit body.
- **`Hmc::has_last_s_full()`** — drop or replace with `std::isnan(last_s_full_)` at the (still nonexistent) Exchange call site when that lands.

## Test plan

- `tests/unit/test_replica.cpp` (or equivalent LLR test): add an assertion that `windowed_.base.last_s_full()` equals a freshly computed `base.s_full(phi)` after each trajectory, on both accept and reject paths.
- `examples/05_su2_llr/`, `examples/06_bose_gas_llr/`, `examples/07_su2_llr/` (whichever currently runs): rerun and confirm `<S − E_n>` time series is bit-identical to pre-change values modulo the elided redundant sweep (which should be bit-identical, since the cached value came from the same call site).
- Force-trigger reject (huge `tau`, large `delta`) and verify `sample()` still returns the correct measurement on rejected steps.

## Estimated diff size

~60 lines added across 5 files. One-commit change once the unit test asserting cache validity is in place.
