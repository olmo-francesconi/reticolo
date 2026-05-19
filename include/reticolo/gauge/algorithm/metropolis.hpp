#pragma once

#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/gauge/concepts.hpp>

#include <cmath>
#include <cstddef>

namespace reticolo::gauge::alg {

struct LinkMetropolisSweep {
    std::size_t accepted = 0;
    std::size_t attempts = 0;

    [[nodiscard]] double acceptance() const noexcept {
        return attempts == 0 ? 0.0 : static_cast<double>(accepted) / static_cast<double>(attempts);
    }
};

// =============================================================================
//  Link Metropolis sweep over a `LinkLocalAction`.
//
//  Per (site, mu): propose theta_new = theta_old + sigma * N(0,1); accept with
//  min(1, exp(-ds_local)). Standard Wilson convention (weight ∝ exp(-S)) —
//  identical to the scalar Metropolis. Visit order is contiguous flat
//  (site-major, then mu) — order is irrelevant for ergodicity on a single
//  thread, and contiguous access keeps the link buffer resident in L1.
//
//  No parity colouring needed for U(1) since two link updates never share
//  state directly (only indirectly through plaquettes — and the staple at
//  link (x, mu) is recomputed each move).
// =============================================================================

template <class A, class R, class F = typename A::value_type>
    requires gauge::LinkLocalAction<A, F> && Rng<R>
class LinkMetropolis {
public:
    using value_type = F;

    LinkMetropolis(A const& action, LinkLattice<F>& field, R& rng, double sigma) noexcept
        : action_{action}, field_{field}, rng_{rng}, sigma_{sigma} {}

    LinkMetropolisSweep sweep() {
        LinkMetropolisSweep stats{};
        std::size_t const d = field_.ndims();
        for (Site x : field_.sites()) {
            for (std::size_t mu = 0; mu < d; ++mu) {
                F const theta_old = field_(x, mu);
                F const theta_new = theta_old + static_cast<F>(sigma_ * rng_.normal());
                auto const ds     = static_cast<double>(action_.ds_local(field_, x, mu, theta_new));
                ++stats.attempts;
                if (ds <= 0.0 || rng_.uniform() < std::exp(-ds)) {
                    field_(x, mu) = theta_new;
                    ++stats.accepted;
                }
            }
        }
        return stats;
    }

    [[nodiscard]] double sigma() const noexcept { return sigma_; }
    void set_sigma(double s) noexcept { sigma_ = s; }

private:
    A const& action_;
    LinkLattice<F>& field_;
    R& rng_;
    double sigma_;
};

}  // namespace reticolo::gauge::alg
