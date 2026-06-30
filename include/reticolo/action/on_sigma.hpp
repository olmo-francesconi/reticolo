#pragma once

#include <reticolo/action/detail/concepts.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace reticolo::action {

// O(N) sigma model. Each site holds a unit N-vector phi(x) ∈ S^(N-1):
//
//   S = -beta * sum_<x,y>  phi(x) · phi(y)
//
// The unit-norm constraint is preserved by `propose`, which draws a uniform
// random vector on the sphere via the Marsaglia normalised-Gaussian trick.
// Apps must seed every site to a unit vector before the first sweep.
//
// HMC on a constrained manifold (RATTLE-style) is deferred — OnSigma
// satisfies LocalAction + HasSEff + HasProposal, which is exactly what
// `alg::Metropolis` needs to drive it.

template <std::size_t N, class T = double>
struct OnSigma {
    using value_type = std::array<T, N>;

    T beta = T{0};

    void describe(log::Entry& e) const {
        e.line("OnSigma<N={}, {}>", N, scalar_name<T>());
        e.param("β={:.3f}", beta);
    }

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
            sum += dot(phi, l[l.next(x, mu)]);
            sum += dot(phi, l[l.prev(x, mu)]);
        }
        return -beta * sum;
    }

    [[nodiscard]] T
    ds_local(Lattice<value_type> const& l, Site x, value_type const& new_v) const noexcept {
        auto const& phi = l[x];
        T delta_s       = T{0};
        for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
            auto const& fwd = l[l.next(x, mu)];
            auto const& bwd = l[l.prev(x, mu)];
            delta_s += dot(new_v, fwd) - dot(phi, fwd);
            delta_s += dot(new_v, bwd) - dot(phi, bwd);
        }
        return -beta * delta_s;
    }

    // Per-site dot in `T`, volume sum accumulated in (and returned as) `double`
    // — see Phi4::s_full for the rationale.
    [[nodiscard]] double s_full(Lattice<value_type> const& l) const noexcept {
        auto const& idx              = l.indexing_ref();
        value_type const* data       = l.data();
        Site::value_type const* next = idx.next_data();
        std::size_t const n          = idx.nsites();
        std::size_t const d          = idx.ndims();

        double total = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            auto const& phi        = data[i];
            std::size_t const base = i * d;
            for (std::size_t mu = 0; mu < d; ++mu) {
                total += static_cast<double>(dot(phi, data[next[base + mu]]));
            }
        }
        double const s = -static_cast<double>(beta) * total;
        last_s_full_   = s;
        return s;
    }

    [[nodiscard]] double last_s_full() const noexcept { return last_s_full_; }
    void restore_last_s_full(double v) const noexcept { last_s_full_ = v; }

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
        // Skip the libm exp on unfavoured bonds (w >= 0 → p = 0).
        if (w >= 0.0) {
            return 0.0;
        }
        return 1.0 - std::exp(w);
    }

    mutable double last_s_full_ = std::numeric_limits<double>::quiet_NaN();

private:
    template <class R>
        requires Rng<R>
    [[nodiscard]] static value_type random_unit_(R& rng) noexcept {
        value_type p{};
        T norm_sq = T{0};
        for (std::size_t i = 0; i < N; ++i) {
            p[i] = static_cast<T>(rng.normal());
            norm_sq += p[i] * p[i];
        }
        T const inv = T{1} / std::sqrt(norm_sq);
        for (std::size_t i = 0; i < N; ++i) {
            p[i] *= inv;
        }
        return p;
    }
};

}  // namespace reticolo::action
