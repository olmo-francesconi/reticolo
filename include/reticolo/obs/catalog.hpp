#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/parallel.hpp>
#include <reticolo/obs/reduce.hpp>

#include <concepts>
#include <cstddef>
#include <stdexcept>

// Observables that are NOT expressible as a per-site kernel — so they can't ride
// `obs::reduce` and get their own function here. Everything that IS a per-site
// quantity (<φ>, <φ²>, |<φ>|, …) is measured with an explicit `obs::reduce` over
// `obs::kernel` (or ad-hoc lambdas) and unpacked with the `*_of` finalizers; there
// are no per-observable wrapper functions.

namespace reticolo::obs {

// Translation-averaged two-point function in direction `mu` at separation `r`:
//  G(r) = (1/N) Σ_x phi(x) phi(x + r·ê_μ)
// Needs a neighbour shift, so it is not a per-site kernel. The shift is periodic,
// so `r` may exceed the linear extent (it wraps). Folded through the canonical
// partition — parallel and thread-count deterministic.
template <std::floating_point T>
[[nodiscard]] double two_point(Lattice<T> const& l, std::size_t r, std::size_t mu) {
    if (mu >= l.ndims()) {
        throw std::out_of_range{"obs::two_point: direction mu out of range"};
    }
    auto const& sh     = l.shape();
    std::size_t stride = 1;
    for (std::size_t d = 0; d < mu; ++d) {
        stride *= sh[d];
    }
    std::size_t const l_mu = sh[mu];
    std::size_t const rr   = r % l_mu;
    T const* const data    = l.data();
    double const sum = exec::field_reduce<double>(l, 1, [&](std::size_t base, std::size_t cnt) {
        double acc = 0.0;
        for (std::size_t j = 0; j < cnt; ++j) {
            std::size_t const i  = base + j;
            std::size_t const c  = (i / stride) % l_mu;
            std::size_t const cn = (c + rr < l_mu) ? (c + rr) : (c + rr - l_mu);
            std::size_t const y  = (i - (c * stride)) + (cn * stride);
            acc += static_cast<double>(data[i]) * static_cast<double>(data[y]);
        }
        return acc;
    });
    return sum / static_cast<double>(l.nsites());
}

}  // namespace reticolo::obs
