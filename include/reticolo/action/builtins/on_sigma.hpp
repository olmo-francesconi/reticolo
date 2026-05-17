#pragma once

#include <reticolo/action/concepts.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <array>
#include <cmath>
#include <cstddef>

namespace reticolo::action {

// =============================================================================
//  O(N) sigma model. Each site holds a unit N-vector phi(x) ∈ S^(N-1):
//
//    S = -beta * sum_<x,y>  phi(x) · phi(y)
//
//  The unit-norm constraint is preserved by `propose`, which draws a uniform
//  random vector on the sphere via the Marsaglia normalised-Gaussian trick.
//  Apps must seed every site to a unit vector before the first sweep.
//
//  HMC on a constrained manifold (RATTLE-style) is deferred — at M8 OnSigma
//  satisfies LocalAction + HasSEff + HasProposal, which is exactly what
//  `alg::Metropolis` needs to drive it.
// =============================================================================

template <std::size_t N>
struct OnSigma {
    using value_type = std::array<double, N>;

    double beta = 0.0;

    [[nodiscard]] static double dot(value_type const& a, value_type const& b) noexcept {
        double s = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            s += a[i] * b[i];
        }
        return s;
    }

    [[nodiscard]] double s_local(Lattice<value_type> const& l, Site x) const noexcept {
        auto const& phi = l[x];
        double sum      = 0.0;
        for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
            Site const fwd = l.next(x, mu);
            Site const bwd = l.prev(x, mu);
            if (fwd.is_valid()) {
                sum += dot(phi, l[fwd]);
            }
            if (bwd.is_valid()) {
                sum += dot(phi, l[bwd]);
            }
        }
        return -beta * sum;
    }

    [[nodiscard]] double
    ds_local(Lattice<value_type> const& l, Site x, value_type const& new_v) const noexcept {
        auto const& phi = l[x];
        double delta_s  = 0.0;
        for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
            Site const fwd = l.next(x, mu);
            Site const bwd = l.prev(x, mu);
            if (fwd.is_valid()) {
                delta_s += dot(new_v, l[fwd]) - dot(phi, l[fwd]);
            }
            if (bwd.is_valid()) {
                delta_s += dot(new_v, l[bwd]) - dot(phi, l[bwd]);
            }
        }
        return -beta * delta_s;
    }

    [[nodiscard]] double s_full(Lattice<value_type> const& l) const noexcept {
        double total = 0.0;
        for (Site const x : l.sites()) {
            auto const& phi = l[x];
            for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
                Site const fwd = l.next(x, mu);
                if (fwd.is_valid()) {
                    total += dot(phi, l[fwd]);
                }
            }
        }
        return -beta * total;
    }

    // Uniform-on-sphere proposal via N-dim Gaussian normalisation. Independent
    // of the current value at x — global proposal works fine for O(N) at all
    // beta because the acceptance ratio just compares neighbour alignment.
    template <class R>
        requires Rng<R>
    [[nodiscard]] value_type
    propose(Lattice<value_type> const& /*l*/, Site /*x*/, R& rng) const noexcept {
        value_type p{};
        double norm_sq = 0.0;
        do {
            norm_sq = 0.0;
            for (std::size_t i = 0; i < N; ++i) {
                p[i] = rng.normal();
                norm_sq += p[i] * p[i];
            }
        } while (norm_sq < 1e-30);
        double const inv = 1.0 / std::sqrt(norm_sq);
        for (std::size_t i = 0; i < N; ++i) {
            p[i] *= inv;
        }
        return p;
    }
};

}  // namespace reticolo::action
