#pragma once

#include <reticolo/action/concepts.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace reticolo::alg {

struct WolffStep {
    std::size_t cluster_size = 0;
};

// =============================================================================
//  Single-cluster Wolff updater for any `WolffEmbeddable` action.
//
//  Algorithm (Wolff 1989):
//    1. Pick a uniformly-random seed site and an action-supplied reflection
//       axis (random unit vector for O(N), random angle for XY).
//    2. BFS the seed's neighbourhood: a candidate link (x in cluster, y not)
//       is activated with action-supplied probability `wolff_link_p` computed
//       on the ORIGINAL field values. y is added to the cluster on activation.
//    3. Once the BFS terminates, reflect every cluster member in-place.
//
//  Storing membership in `in_cluster_` and deferring the flip to the end keeps
//  link probabilities consistent (every test sees the same pre-flip field) and
//  avoids the order-dependence trap of flipping on visit.
//
//  Open-BC: invalid neighbours (returned as `Site::k_invalid_value` by the
//  `Indexing`) are skipped, so the cluster naturally terminates at boundaries.
// =============================================================================
template <class A, class R, class F = typename A::value_type>
    requires action::WolffEmbeddable<A, F, R>
class Wolff {
public:
    using value_type = F;
    using axis_type  = typename A::axis_type;

    Wolff(A const& action, Lattice<F>& field, R& rng)
        : action_{action}, field_{field}, rng_{rng}, in_cluster_(field.nsites(), 0) {
        stack_.reserve(field.nsites());
    }

    WolffStep update() {
        std::size_t const n_sites = field_.nsites();
        Site const seed{rng_.uniform_int(n_sites)};
        axis_type const axis = action_.wolff_random_axis(rng_);

        std::ranges::fill(in_cluster_, std::uint8_t{0});
        stack_.clear();
        stack_.push_back(seed);
        in_cluster_[seed.value()] = 1;

        std::size_t cluster_size = 0;
        while (!stack_.empty()) {
            Site const x = stack_.back();
            stack_.pop_back();
            ++cluster_size;
            for (std::size_t mu = 0; mu < field_.ndims(); ++mu) {
                consider_link_(x, field_.next(x, mu), axis);
                consider_link_(x, field_.prev(x, mu), axis);
            }
        }

        for (std::size_t i = 0; i < n_sites; ++i) {
            if (in_cluster_[i] != 0) {
                Site const s{i};
                field_[s] = action_.wolff_reflect(field_[s], axis);
            }
        }

        return {.cluster_size = cluster_size};
    }

private:
    void consider_link_(Site x, Site y, axis_type const& axis) {
        if (!y.is_valid()) {
            return;
        }
        if (in_cluster_[y.value()] != 0) {
            return;
        }
        double const p = action_.wolff_link_p(field_[x], field_[y], axis);
        if (p > 0.0 && rng_.uniform() < p) {
            in_cluster_[y.value()] = 1;
            stack_.push_back(y);
        }
    }

    A const& action_;
    Lattice<F>& field_;
    R& rng_;
    std::vector<std::uint8_t> in_cluster_;
    std::vector<Site> stack_;
};

}  // namespace reticolo::alg
