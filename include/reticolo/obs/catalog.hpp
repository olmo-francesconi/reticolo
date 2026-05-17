#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/site.hpp>

#include <array>
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

// m2 = <phi^2> = (1/V) Σ_x phi(x)^2.  Aliased separately from `sq` for naming
// clarity. NOTE: this is the per-site field squared, NOT the squared
// magnetization (Σ phi / V)^2 — use `mean_sq` for the latter. Both quantities
// are commonly written "m^2" in different parts of the literature.
template <class T>
[[nodiscard]] double m2(Lattice<T> const& l) noexcept {
    return sq(l);
}

// (Σ_x phi(x) / V)^2 — the SQUARED magnetization-per-site of one configuration.
// This is the right input for the connected susceptibility
//   chi = V * (<mean_sq> - <|mean|>^2)
// and for the Binder cumulant of a scalar field. Distinct from `m2` above.
template <class T>
[[nodiscard]] double mean_sq(Lattice<T> const& l) noexcept {
    double const m = mean(l);
    return m * m;
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

// Squared magnetization per site for a vector-valued field:
//   |M|^2 / V^2  with M = Σ_x phi(x)  (phi(x) ∈ R^N)
// Returns a rotation-invariant scalar suitable for ensemble averaging into
// susceptibility / Binder cumulants of an O(N) symmetry.
template <std::size_t N>
[[nodiscard]] double vector_magnetization_sq(Lattice<std::array<double, N>> const& l) noexcept {
    std::array<double, N> sum{};
    for (Site const x : l.sites()) {
        auto const& v = l[x];
        for (std::size_t i = 0; i < N; ++i) {
            sum[i] += v[i];
        }
    }
    double m_sq = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        m_sq += sum[i] * sum[i];
    }
    auto const inv_v_sq = 1.0 / (static_cast<double>(l.nsites()) * static_cast<double>(l.nsites()));
    return m_sq * inv_v_sq;
}

// XY-specialised |M|^2 / V^2 for a Lattice<theta>: projects theta -> (cosθ,sinθ)
// before summing so the rotation-invariant 2-vector magnetization comes out
// without forcing the app to allocate an array<double,2> field.
template <class T>
[[nodiscard]] double xy_magnetization_sq(Lattice<T> const& theta) noexcept {
    double mx = 0.0;
    double my = 0.0;
    for (Site const x : theta.sites()) {
        auto const t = static_cast<double>(theta[x]);
        mx += std::cos(t);
        my += std::sin(t);
    }
    auto const inv_v_sq =
        1.0 / (static_cast<double>(theta.nsites()) * static_cast<double>(theta.nsites()));
    return ((mx * mx) + (my * my)) * inv_v_sq;
}

}  // namespace reticolo::obs
