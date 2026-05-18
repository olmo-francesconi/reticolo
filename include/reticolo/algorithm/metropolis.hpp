#pragma once

#include <reticolo/action/concepts.hpp>
#include <reticolo/action/hot_loop.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <cmath>
#include <cstddef>

namespace reticolo::alg {

struct MetropolisSweep {
    std::size_t accepted = 0;
    std::size_t attempts = 0;

    [[nodiscard]] double acceptance() const noexcept {
        return attempts == 0 ? 0.0 : static_cast<double>(accepted) / static_cast<double>(attempts);
    }
};

// =============================================================================
//  Metropolis sweep over a `LocalAction`.
//
//  Per site:
//    * Propose new_v. If the action satisfies `HasProposal`, the action's
//      `propose(field, x, rng)` is used. Otherwise we fall back to a Gaussian
//      increment of width `sigma`.
//    * Accept with probability min(1, exp(-ds_local)).
//
//  Visit order: contiguous flat-index iota — order is irrelevant for
//  ergodicity on a single thread, and contiguous access keeps the field
//  resident in L1.
// =============================================================================
template <class A, class R, class F = typename A::value_type>
    requires action::LocalAction<A, F> && Rng<R>
class Metropolis {
public:
    using value_type = F;

    Metropolis(A const& action, Lattice<F>& field, R& rng, double sigma) noexcept
        : action_{action}, field_{field}, rng_{rng}, sigma_{sigma} {}

    MetropolisSweep sweep() {
        MetropolisSweep stats{};
        if constexpr (action::HasDsLocalFromNbrs<A, F>) {
            // Fast path: visit_nn computes nbrs with direct stride offsets and
            // hands (phi, nbrs) to the body. The body is sequential — each
            // accepted write is visible to the next site's neighbour sum.
            F* const fdata = field_.data();
            action::detail::visit_nn<F>(
                field_, [this, fdata, &stats](std::size_t i, F phi, F nbrs) {
                    F const new_v = propose_local_(phi, i);
                    auto const ds =
                        static_cast<double>(action_.ds_local_from_nbrs(phi, new_v, nbrs));
                    ++stats.attempts;
                    if (ds <= 0.0 || rng_.uniform() < std::exp(-ds)) {
                        fdata[i] = new_v;
                        ++stats.accepted;
                    }
                });
        } else {
            for (Site x : field_.sites()) {
                F const new_v = propose_(x);
                auto const ds = static_cast<double>(action_.ds_local(field_, x, new_v));
                ++stats.attempts;
                if (ds <= 0.0 || rng_.uniform() < std::exp(-ds)) {
                    field_[x] = new_v;
                    ++stats.accepted;
                }
            }
        }
        return stats;
    }

    [[nodiscard]] double sigma() const noexcept { return sigma_; }
    void set_sigma(double s) noexcept { sigma_ = s; }

private:
    [[nodiscard]] F propose_(Site x) {
        if constexpr (action::HasProposal<A, F, R>) {
            return action_.propose(field_, x, rng_);
        } else {
            return field_[x] + static_cast<F>(sigma_ * rng_.normal());
        }
    }

    // Variant for the fast path that already has phi in a register and so can skip
    // re-reading from the lattice. Custom proposals still go through the action.
    [[nodiscard]] F propose_local_(F phi, std::size_t i) {
        if constexpr (action::HasProposal<A, F, R>) {
            return action_.propose(field_, Site{i}, rng_);
        } else {
            return phi + static_cast<F>(sigma_ * rng_.normal());
        }
    }

    A const& action_;
    Lattice<F>& field_;
    R& rng_;
    double sigma_;
};

}  // namespace reticolo::alg
