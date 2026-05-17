#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/site.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>

// =============================================================================
//  Per-configuration observers for scalar lattices.
//
//  Every observer returns a `double` so series storage is uniform. The
//  ensemble-level reductions (susceptibility, Binder cumulant, etc.) live in
//  reticolo/obs/analysis.hpp; they take a span of measurements.
// =============================================================================

namespace reticolo::obs {

// <phi> = (1/N) Σ_x phi(x).
template <class T>
[[nodiscard]] double mean(Lattice<T> const& l) noexcept {
    double sum = 0.0;
    for (Site const x : l.sites()) {
        sum += static_cast<double>(l[x]);
    }
    return sum / static_cast<double>(l.nsites());
}

// <phi^2> = (1/N) Σ_x phi(x)^2.  Same as obs::m2.
template <class T>
[[nodiscard]] double sq(Lattice<T> const& l) noexcept {
    double sum = 0.0;
    for (Site const x : l.sites()) {
        auto const v = static_cast<double>(l[x]);
        sum += v * v;
    }
    return sum / static_cast<double>(l.nsites());
}

// Magnetisation per site: |m| where m = (1/N) Σ phi(x). Useful in broken-
// symmetry phases where the sign of m fluctuates between configurations.
template <class T>
[[nodiscard]] double magnetization(Lattice<T> const& l) noexcept {
    return std::abs(mean(l));
}

// m2 = <phi^2>.  Aliased separately from `sq` for naming clarity when an app
// records both for downstream susceptibility / Binder cumulant computation.
template <class T>
[[nodiscard]] double m2(Lattice<T> const& l) noexcept {
    return sq(l);
}

// m4 = (1/N) Σ_x phi(x)^4.
template <class T>
[[nodiscard]] double m4(Lattice<T> const& l) noexcept {
    double sum = 0.0;
    for (Site const x : l.sites()) {
        auto const v  = static_cast<double>(l[x]);
        auto const v2 = v * v;
        sum += v2 * v2;
    }
    return sum / static_cast<double>(l.nsites());
}

// Translation-averaged two-point function in direction `mu` at separation `r`:
//   G(r) = (1/N) Σ_x phi(x) phi(x + r·ê_μ)
// where the shift uses the lattice's neighbour table (`r` applications of next).
// Requires the lattice to be periodic along `mu` for `r > 0`; throws otherwise.
template <class T>
[[nodiscard]] double two_point(Lattice<T> const& l, std::size_t r, std::size_t mu) {
    if (mu >= l.ndims()) {
        throw std::out_of_range{"obs::two_point: direction mu out of range"};
    }
    if (r > 0 && l.bcs().affects_topology(mu)) {
        throw std::invalid_argument{
            "obs::two_point: open boundary in direction mu would yield invalid shifts"};
    }
    double sum = 0.0;
    for (Site const x : l.sites()) {
        Site y = x;
        for (std::size_t step = 0; step < r; ++step) {
            y = l.next(y, mu);
        }
        sum += static_cast<double>(l[x]) * static_cast<double>(l[y]);
    }
    return sum / static_cast<double>(l.nsites());
}

}  // namespace reticolo::obs
