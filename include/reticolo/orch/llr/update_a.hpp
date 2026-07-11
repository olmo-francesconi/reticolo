#pragma once

namespace reticolo::orch::llr {

// Updates for the LLR slope `a` of a single replica.
//
// Derivation (Gaussian-penalty window):
//    The tilted, windowed log-density is
//      psi(E) = ln g(E) - (1+a) E - (E - E_n)^2 / (2 delta^2)
//    Linearising near the peak with a = d ln g/dE | _{E_n} - 1 (fixed point):
//      <E - E_n>(a)  ≈  delta^2 * ( a* - a )
//    so a single Newton step is
//      a  <-  a + <E - E_n> / delta^2
//
// Note: the often-quoted "12 / delta^2" coefficient is for the HARD-window
// case where the window function is the indicator on an interval of width
// delta (variance delta^2/12). For the Gaussian penalty `(E - E_n)^2 / (2*
// delta^2)` used here the variance is delta^2, so the coefficient is 1.
// Mixing the two gives a Newton step that is 12x too aggressive and drives
// the slopes to wildly divergent values.
//
// Robbins-Monro: same step damped by 1/(k+1). Counter k restarts at 0 after
// the NR warm-up.

template <class T>
// NOLINTNEXTLINE(readability-identifier-naming) physics convention: mean_dE = <E - E_n>
[[nodiscard]] T nr_update(T a, T mean_dE, T delta) noexcept {
    return a + (mean_dE / (delta * delta));
}

template <class T>
// NOLINTNEXTLINE(readability-identifier-naming) physics convention: mean_dE = <E - E_n>
[[nodiscard]] T rm_update(T a, T mean_dE, T delta, int k) noexcept {
    return a + (mean_dE / (delta * delta * static_cast<T>(k + 1)));
}

}  // namespace reticolo::orch::llr
