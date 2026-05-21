#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/site.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

// Per-configuration observers for scalar lattices.
//
// Every observer returns a `double` so series storage is uniform. The
// ensemble-level reductions (susceptibility, Binder cumulant, etc.) live in
// reticolo/obs/analysis.hpp; they take a span of measurements.
//
// Naming:
//   obs::mean       — <φ>           = (1/V) Σ_x φ(x)
//   obs::sq         — <φ²>          = (1/V) Σ_x φ(x)²
//   obs::quartic    — <φ⁴>          = (1/V) Σ_x φ(x)⁴
//   obs::sq_of_mean — <φ>²          = ((1/V) Σ_x φ(x))²
//   obs::two_point  — G(r,μ)        translation-averaged two-point function
//   obs::mag::abs   — |<φ>|         magnetisation modulus for a scalar field
//   obs::mag::xy_sq — |M|²/V²       for a θ-valued (XY) lattice
//   obs::mag::on_sq — |M|²/V²       for an O(N) array<T,N> lattice
//
// Moments live bare in `obs::`. Anything specific to the magnetisation
// symmetry channel (scalar / XY / O(N)) lives under `obs::mag::`.

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

// <phi^2> = (1/N) Σ_x phi(x)^2.
template <class T>
[[nodiscard]] double sq(Lattice<T> const& l) noexcept {
    double sum = 0.0;
    for (Site const x : l.sites()) {
        auto const v = static_cast<double>(l[x]);
        sum += v * v;
    }
    return sum / static_cast<double>(l.nsites());
}

// <phi^4> = (1/N) Σ_x phi(x)^4.
template <class T>
[[nodiscard]] double quartic(Lattice<T> const& l) noexcept {
    double sum = 0.0;
    for (Site const x : l.sites()) {
        auto const v  = static_cast<double>(l[x]);
        auto const v2 = v * v;
        sum += v2 * v2;
    }
    return sum / static_cast<double>(l.nsites());
}

// (Σ_x phi(x) / V)^2 — the squared magnetisation-per-site of one configuration.
// Right input for the connected susceptibility
//  chi = V * (<sq_of_mean> - <mag::abs>^2)
// and for the Binder cumulant of a scalar field. Distinct from `sq` (= <φ²>),
// which is the per-site field squared.
template <class T>
[[nodiscard]] double sq_of_mean(Lattice<T> const& l) noexcept {
    double const m = mean(l);
    return m * m;
}

// Translation-averaged two-point function in direction `mu` at separation `r`:
//  G(r) = (1/N) Σ_x phi(x) phi(x + r·ê_μ)
// where the shift uses the lattice's neighbour table (`r` applications of next).
// The lattice is always periodic, so `r` may exceed the linear extent without
// concern — wraps just keep walking.
template <class T>
[[nodiscard]] double two_point(Lattice<T> const& l, std::size_t r, std::size_t mu) {
    if (mu >= l.ndims()) {
        throw std::out_of_range{"obs::two_point: direction mu out of range"};
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

namespace mag {

// Magnetisation per site: |m| where m = (1/N) Σ phi(x). Useful in broken-
// symmetry phases where the sign of m fluctuates between configurations.
template <class T>
[[nodiscard]] double abs(Lattice<T> const& l) noexcept {
    return std::abs(obs::mean(l));
}

// Squared magnetisation per site for a vector-valued field:
//  |M|^2 / V^2  with M = Σ_x phi(x)  (phi(x) ∈ R^N)
// Returns a rotation-invariant scalar suitable for ensemble averaging into
// susceptibility / Binder cumulants of an O(N) symmetry.
template <std::size_t N, class T>
[[nodiscard]] double on_sq(Lattice<std::array<T, N>> const& l) noexcept {
    std::array<double, N> sum{};
    for (Site const x : l.sites()) {
        auto const& v = l[x];
        for (std::size_t i = 0; i < N; ++i) {
            sum[i] += static_cast<double>(v[i]);
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
// before summing so the rotation-invariant 2-vector magnetisation comes out
// without forcing the app to allocate an array<double,2> field.
template <class T>
[[nodiscard]] double xy_sq(Lattice<T> const& theta) noexcept {
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

}  // namespace mag

}  // namespace reticolo::obs
