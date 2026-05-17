#pragma once

#include <reticolo/action/concepts.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <algorithm>
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

template <std::size_t N, class T = double>
struct OnSigma {
    using value_type = std::array<T, N>;

    T beta = T{0};

    [[nodiscard]] static T dot(value_type const& a, value_type const& b) noexcept {
        T s = T{0};
        for (std::size_t i = 0; i < N; ++i) {
            s += a[i] * b[i];
        }
        return s;
    }

    [[nodiscard]] T s_local(Lattice<value_type> const& l, Site x) const noexcept {
        auto const& phi = l[x];
        T sum           = T{0};
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

    [[nodiscard]] T
    ds_local(Lattice<value_type> const& l, Site x, value_type const& new_v) const noexcept {
        auto const& phi = l[x];
        T delta_s       = T{0};
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

    [[nodiscard]] T s_full(Lattice<value_type> const& l) const noexcept {
        T total = T{0};
        for (Site const x : l.bulk_sites()) {
            auto const& phi = l[x];
            for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
                total += dot(phi, l[l.next(x, mu)]);
            }
        }
        for (Site const x : l.skin_sites()) {
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
        return random_unit_(rng);
    }

    // -------- Wolff embedding ------------------------------------------------
    //
    // The reflection axis is a uniformly-random unit vector r on S^(N-1); the
    // reflection across the plane perpendicular to r is the Householder map
    // R(phi) = phi - 2(r·phi)r. The bond-activation probability is
    //
    //   p = 1 - exp(min(0, -2*beta*(r·phi_x)(r·phi_y)))
    //
    // which is Wolff (1989) eq. (4) for O(N).
    using axis_type = value_type;

    template <class R>
        requires Rng<R>
    [[nodiscard]] axis_type wolff_random_axis(R& rng) const noexcept {
        return random_unit_(rng);
    }

    [[nodiscard]] value_type wolff_reflect(value_type const& phi,
                                           axis_type const& r) const noexcept {
        T const proj = T{2} * dot(phi, r);
        value_type out{};
        for (std::size_t i = 0; i < N; ++i) {
            out[i] = phi[i] - proj * r[i];
        }
        return out;
    }

    [[nodiscard]] double wolff_link_p(value_type const& phi_x,
                                      value_type const& phi_y,
                                      axis_type const& r) const noexcept {
        double const w = -2.0 * static_cast<double>(beta) * static_cast<double>(dot(r, phi_x)) *
                         static_cast<double>(dot(r, phi_y));
        return 1.0 - std::exp(std::min(0.0, w));
    }

private:
    template <class R>
        requires Rng<R>
    [[nodiscard]] static value_type random_unit_(R& rng) noexcept {
        value_type p{};
        T norm_sq = T{0};
        do {
            norm_sq = T{0};
            for (std::size_t i = 0; i < N; ++i) {
                p[i] = static_cast<T>(rng.normal());
                norm_sq += p[i] * p[i];
            }
        } while (static_cast<double>(norm_sq) < 1e-30);
        T const inv = T{1} / std::sqrt(norm_sq);
        for (std::size_t i = 0; i < N; ++i) {
            p[i] *= inv;
        }
        return p;
    }
};

}  // namespace reticolo::action
