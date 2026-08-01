#pragma once

#include <cmath>

namespace reticolo::orch::llr {

// Deep-window warm-in schedule — shared by the CPU driver
// (`orch::llr::Replica::warm_into_window`) and the CUDA driver
// (`cuda::llr::warm_all`).
//
// A fixed-δ window is stiff: slamming it onto a field that thermalized far from
// E_n (|E_n − ⟨Q⟩_base| ≫ δ) produces a force ∝ |Q − E_n|/δ² that overflows the
// integrator's stable step and NaNs the trajectory. The cure is to walk the
// window CENTRE from the field's current constraint value out to E_n in steps of
// at most ½δ, adapting the tilt each step so the equilibrium follows — the field
// then stays within a window width of the centre throughout and the force never
// spikes.
//
// This is a §2.4 category-3 quantity: a schedule that is not per-site math but
// whose absence changes results (it is the difference between a seated replica
// and a NaN). It lives here, single-sourced and parameterised, rather than being
// written once per backend — which is exactly how the CUDA driver came to lack
// it after `8e693e8` added it to the CPU side.
//
// Templated on the replica scalar so each caller keeps its own arithmetic: an
// f32 replica must not have its centres routed through double and cast back, or
// the bit-exact resume guarantee breaks.

// Number of hops from the field's current value to the true window centre.
// `gap = E_n − q0`. The loop runs k = 1…n with centre `ramp_centre(q0, gap, k,
// n)`, so k = n lands exactly on E_n: these are HOPS, not interior points, and
// covering `gap` in hops of at most ½δ needs ceil(|gap| / (½δ)) of them.
//
// Returns 0 — seat directly, no ramp — when the field already sits within ½δ of
// the centre, or when δ is degenerate.
//
// (Through `8e693e8` this subtracted 1, which is the interior-point count and
// gave one hop too few: gap=5, δ=2 produced 4 hops of 1.25 against an intended
// 1.0, and gap ∈ (½δ, δ] produced a single hop across the whole gap. Corrected
// deliberately — it changes every LLR warm-up, hence every chain.)
template <class T>
[[nodiscard]] inline int ramp_steps(T gap, T delta) noexcept {
    if (!(delta > T{0})) {
        return 0;
    }
    double const need = std::abs(static_cast<double>(gap)) / (0.5 * static_cast<double>(delta));
    int const n       = static_cast<int>(std::ceil(need));
    return n > 1 ? n : 0;
}

// The k-th intermediate centre, k ∈ [1, n_steps]. k == n_steps lands exactly on
// the true centre; callers restore the true centre afterwards regardless.
template <class T>
[[nodiscard]] inline T ramp_centre(T q0, T gap, int k, int n_steps) noexcept {
    return q0 + ((gap * static_cast<T>(k)) / static_cast<T>(n_steps));
}

}  // namespace reticolo::orch::llr
