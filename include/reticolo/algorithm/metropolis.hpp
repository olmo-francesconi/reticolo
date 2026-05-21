#pragma once

#include <reticolo/action/detail/concepts.hpp>
#include <reticolo/action/detail/helpers.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/log_helpers.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <cmath>
#include <cstddef>

namespace reticolo::alg {

struct MetropolisSpec {
    double sigma = 1.0;
};

struct MetropolisResult {
    std::size_t accepted = 0;
    std::size_t attempts = 0;

    [[nodiscard]] double acceptance() const noexcept {
        return attempts == 0 ? 0.0 : static_cast<double>(accepted) / static_cast<double>(attempts);
    }
};

// =============================================================================
//  Metropolis sweep — one class for both site fields (Lattice<F>) and link
//  fields (LinkLattice<F>). Three code paths picked at compile time:
//
//    1. HasDsLocalFromNbrs (scalar action with NN-only ds_local):
//       visit_nn fast path with direct-stride neighbour reads; the
//       acceptance body reuses the in-register `phi` instead of re-reading.
//
//    2. HasProposal (action provides its own proposal kernel):
//       proposal comes from action_.propose(field, loc..., rng); fallback
//       is Gaussian. Hooked into both the fast and generic paths.
//
//    3. Generic field.for_each_update visit (works for Lattice via
//       (T&, Site) and LinkLattice via (T&, Site, mu)). The body's
//       parameter pack `loc...` absorbs whichever extra coords the field
//       yields, and `action_.ds_local(field, loc..., new_v)` unpacks the
//       matching arity.
// =============================================================================

template <class A, class R, class F = typename A::value_type, class Field = Lattice<F>>
    requires(action::LocalAction<A, F> || gauge::LinkLocalAction<A, F>) && Rng<R>
class Metropolis {
public:
    using value_type = F;

    static constexpr std::string_view log_tag = "metr";

    Metropolis(A const& action,
               Field& field,
               R& rng,
               MetropolisSpec const& spec,
               log::Mode announce = log::Mode::normal)
        : action_{action}, field_{field}, rng_{rng}, sigma_{spec.sigma} {
        if (announce == log::Mode::normal) {
            log::algo(*this);
        }
    }

    void describe(log::Entry& e) const {
        e.line("Metropolis");
        e.param("σ={:.3f}", sigma_);
    }

    MetropolisResult step(log::Mode log_mode = log::Mode::normal) {
        MetropolisResult stats{};

        if constexpr (action::HasDsLocalFromNbrs<A, F>) {
            // Scalar fast path: visit_nn precomputes the NN sum and the body
            // accepts (phi, nbrs) directly. Bit-stable order, vectorised
            // neighbour sum.
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
            ++step_count_;
            if (log_mode == log::Mode::normal) {
                log::info("metr", "sweep {:>6}  acc={:.3f}", step_count_, stats.acceptance());
            }
            return stats;
        }

        // Generic path. The trailing `auto... loc` pack captures the field's
        // location coords: () empty for Lattice (already absorbed by ref) — wait,
        // not quite: Lattice yields (ref, Site), so loc = (Site). LinkLattice
        // yields (ref, Site, mu), so loc = (Site, mu). Either way the action's
        // existing ds_local signature unpacks the right arity.
        field_.for_each_update([&](F& ref, auto... loc) {
            F const old   = ref;
            F const new_v = propose_(old, loc...);
            auto const ds = static_cast<double>(action_.ds_local(field_, loc..., new_v));
            ++stats.attempts;
            if (ds <= 0.0 || rng_.uniform() < std::exp(-ds)) {
                ref = new_v;
                ++stats.accepted;
            }
        });
        ++step_count_;
        if (log_mode == log::Mode::normal) {
            log::info("metr", "sweep {:>6}  acc={:.3f}", step_count_, stats.acceptance());
        }
        return stats;
    }

    [[nodiscard]] double sigma() const noexcept { return sigma_; }
    void set_sigma(double s) noexcept { sigma_ = s; }

private:
    template <class... Loc>
    [[nodiscard]] F propose_(F old, Loc... loc) {
        if constexpr (action::HasProposal<A, F, R>) {
            return action_.propose(field_, loc..., rng_);
        } else {
            return old + static_cast<F>(sigma_ * rng_.normal());
        }
    }

    // Variant for the visit_nn fast path that already has phi in a register
    // and so can skip re-reading from the lattice. Custom proposals still
    // route through the action.
    [[nodiscard]] F propose_local_(F phi, std::size_t i) {
        if constexpr (action::HasProposal<A, F, R>) {
            return action_.propose(field_, Site{i}, rng_);
        } else {
            return phi + static_cast<F>(sigma_ * rng_.normal());
        }
    }

    A const& action_;
    Field& field_;
    R& rng_;
    double sigma_;

    // Cumulative sweep counter — advances on every call, silent or not.
    std::size_t step_count_ = 0;
};

}  // namespace reticolo::alg
