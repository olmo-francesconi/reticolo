#pragma once

#include <reticolo/action/detail/concepts.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/log_helpers.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace reticolo::alg {

struct WolffSpec {};

struct WolffResult {
    std::size_t cluster_size = 0;

    // Uniform with HmcResult / MetropolisResult: every cluster flip is accepted
    // by construction, so the per-update acceptance is identically 1.
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    [[nodiscard]] double acceptance() const noexcept { return 1.0; }
};

// =============================================================================
//  Single-cluster Wolff updater for any `WolffEmbeddable` action.
//
//  Algorithm (Wolff 1989):
//    1. Pick a uniformly-random seed site and an action-supplied reflection
//       axis (random unit vector for O(N), random angle for XY).
//    2. DFS the seed's neighbourhood (LIFO via `stack_`): a candidate link
//       (x in cluster, y not) is activated with action-supplied probability
//       `wolff_link_p` computed on the ORIGINAL field values. y is added to
//       the cluster on activation.
//    3. Once the walk terminates, reflect every cluster member in-place.
//
//  Membership uses a per-call generation tag (`gen_`) compared against
//  `mark_[i]` — bumping `gen_` is O(1) and effectively clears the membership
//  set, so the clear cost does not scale with `nsites()` per update. When the
//  tag is about to wrap, the buffer is zeroed and `gen_` reset.
//
//  Deferring the flip to the end keeps link probabilities consistent (every
//  test sees the same pre-flip field) and avoids the order-dependence trap
//  of flipping on visit.
// =============================================================================
template <class A, class R, class F = typename A::value_type>
    requires action::WolffEmbeddable<A, F, R>
class Wolff {
public:
    using value_type = F;
    using axis_type  = typename A::axis_type;
    using mark_type  = std::uint32_t;

    static constexpr std::string_view log_tag = "wolf";

    Wolff(A const& action,
          Lattice<F>& field,
          R& rng,
          WolffSpec const& /*spec*/ = {},
          log::Mode announce        = log::Mode::normal)
        : action_{action}, field_{field}, rng_{rng}, mark_(field.nsites(), 0) {
        stack_.reserve(field.nsites());
        cluster_sites_.reserve(field.nsites());
        if (announce == log::Mode::normal) {
            log::algo(*this);
        }
    }

    void describe(log::Entry& e) const { e.line("Wolff"); }

    WolffResult step(log::Mode log_mode = log::Mode::normal) {
        std::size_t const n_sites = field_.nsites();
        Site const seed{rng_.uniform_int(n_sites)};
        axis_type const axis = action_.wolff_random_axis(rng_);

        bump_generation_();
        stack_.clear();
        cluster_sites_.clear();
        stack_.push_back(seed);
        cluster_sites_.push_back(seed);
        mark_[seed.value()] = gen_;

        while (!stack_.empty()) {
            Site const x = stack_.back();
            stack_.pop_back();
            for (std::size_t mu = 0; mu < field_.ndims(); ++mu) {
                consider_link_(x, field_.next(x, mu), axis);
                consider_link_(x, field_.prev(x, mu), axis);
            }
        }

        for (Site const s : cluster_sites_) {
            field_[s] = action_.wolff_reflect(field_[s], axis);
        }

        ++step_count_;
        if (log_mode == log::Mode::normal) {
            log::info("wolf", "update {:>6}  cluster={}", step_count_, cluster_sites_.size());
        }
        return {.cluster_size = cluster_sites_.size()};
    }

private:
    void consider_link_(Site x, Site y, axis_type const& axis) {
        if (mark_[y.value()] == gen_) {
            return;
        }
        double const p = action_.wolff_link_p(field_[x], field_[y], axis);
        if (p > 0.0 && rng_.uniform() < p) {
            mark_[y.value()] = gen_;
            cluster_sites_.push_back(y);
            stack_.push_back(y);
        }
    }

    void bump_generation_() {
        if (gen_ == std::numeric_limits<mark_type>::max()) {
            std::fill(mark_.begin(), mark_.end(), mark_type{0});
            gen_ = 0;
        }
        ++gen_;
    }

    A const& action_;
    Lattice<F>& field_;
    R& rng_;
    std::vector<mark_type> mark_;
    std::vector<Site> stack_;
    std::vector<Site> cluster_sites_;
    mark_type gen_ = 0;

    // Cumulative update counter — advances on every call, silent or not.
    std::size_t step_count_ = 0;
};

}  // namespace reticolo::alg
