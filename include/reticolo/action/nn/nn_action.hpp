#pragma once

#include <reticolo/action/cache.hpp>
#include <reticolo/core/exec/nn_stencil.hpp>
#include <reticolo/core/exec/parallel.hpp>
#include <reticolo/core/field/lattice.hpp>

#include <cstddef>
#include <utility>
#include <vector>

// NNAction<Derived, T> — the unified nearest-neighbour scalar action base. Every
// NN action is the SAME lattice traversal: for each site, fold a per-bond COMBINE
// over the neighbours into `agg`, then FINALIZE (self, agg) into the per-site
// value. Actions differ only in those two operations (plus an optional overall
// scale). It subsumes the former SiteAction and BondAction:
//
//   * site actions (Phi4/Phi6/SineGordon): combine = identity (agg = Σ neighbour
//     values), finalize = the leaf's per-site kernel of (phi, Σnbrs).
//   * bond actions (XY): combine = the per-bond function (e.g. cos(self−nbr)),
//     finalize = pass the aggregate through, with `action_scale` (e.g. −β)
//     applied to the TOTAL — kept a post-reduce multiply so the float summation
//     order (hence s_full to the last bit) is unchanged from the old BondAction.
//
// The leaf supplies the two finalizers:
//   auto action_kernel() const;   // (T self, T fwd_agg)        -> T   total-S site
//   auto force_kernel()  const;   // (size_t i, T self, T agg)  -> T   force site
// and OPTIONALLY (defaults in parens):
//   auto action_combine() const;  // (T self, T nbr) -> T   [identity]
//   auto force_combine()  const;  // (T self, T nbr) -> T   [identity]
//   T    action_scale()   const;  // total-S prefactor       [T{1}]
//   void prep(Lattice<T> const&) const;   // fill scratch() before a force pass
//                                         // (e.g. SineGordon's sin/cos batch)
//
// Both finalizers/combines return a callable capturing the couplings by value, so
// the base hoists them into the hot loop exactly as a hand-written action would —
// the vectorised inner loop is unchanged. A leaf whose s_full doesn't fit the
// plain finalize form (SineGordon's batched transcendental) declares its own
// s_full, hiding the base one; the shared cache still lives here. The LLR fused
// `s_full_and_force` fast-path is opt-in via `staged_force_energy`.

namespace reticolo::action {

template <class Derived, class T>
struct NNAction : SFullCache {
    [[nodiscard]] double s_full(Lattice<T> const& l) const noexcept {
        auto comb        = action_combine_();
        auto fin         = derived_().action_kernel();
        double const raw = exec::nn_reduce<exec::FwdOnly, T, double>(
            l, comb, [&fin](T self, T agg) { return static_cast<double>(fin(self, agg)); });
        double const s = static_cast<double>(action_scale_()) * raw;
        last_s_full_   = s;
        return s;
    }

    void compute_force(Lattice<T> const& l, Lattice<T>& force) const noexcept {
        maybe_prep_(l);
        auto comb    = force_combine_();
        auto fin     = derived_().force_kernel();
        T* const out = force.data();
        // nn_visit self-threads via the parallel primitives (write-disjoint).
        exec::nn_visit<exec::AllDirs, T>(
            l, comb, [&fin, out](std::size_t i, T self, T agg) { out[i] = fin(i, self, agg); });
    }

    void compute_force_and_kick(Lattice<T> const& l, Lattice<T>& mom, T k_dt) const noexcept {
        maybe_prep_(l);
        auto comb  = force_combine_();
        auto fin   = derived_().force_kernel();
        T* const m = mom.data();
        exec::nn_visit<exec::AllDirs, T>(l, comb, [&fin, m, k_dt](std::size_t i, T self, T agg) {
            m[i] += k_dt * fin(i, self, agg);
        });
    }

    // Fused total action + force in ONE AllDirs pass, driven by a leaf's
    // force-energy kernel `(i, self, agg) -> {force, s_site}` (site-style leaves:
    // the aggregate is the full 2·d neighbour sum). Per-site action values are
    // staged in scratch and reduced in a separate linear pass — accumulating into
    // a scalar inside the visit would put a loop-carried FP dependency in the
    // force loop and serialise it. Does not touch the `last_s_full` cache.
    template <class Kernel>
    [[nodiscard]] double
    staged_force_energy(Lattice<T> const& l, Lattice<T>& force, Kernel&& kern) const noexcept {
        T* const out        = force.data();
        std::size_t const n = l.nsites();
        ensure_scratch(n);
        T* const sb = scratch_.data();
        exec::nn_visit<exec::AllDirs, T>(
            l, exec::IdentityCombine{}, [&kern, out, sb](std::size_t i, T self, T agg) {
                auto const [f, s] = kern(i, self, agg);
                out[i]            = f;
                sb[i]             = s;
            });
        return reticolo::exec::field_reduce(l, 1, [sb](std::size_t base, std::size_t cnt) {
            return reticolo::exec::lane_sum8(
                base, cnt, [sb](std::size_t i) { return static_cast<double>(sb[i]); });
        });
    }

    // Lazy per-site scratch, sized to nsites — shared by the LLR fast-path and by
    // any leaf `prep`/`s_full` that batches a transcendental (SineGordon). Public
    // (like the cache) to keep the derived struct an aggregate.
    void ensure_scratch(std::size_t n) const {
        if (scratch_.size() < n) {
            scratch_.resize(n);
        }
    }

    mutable std::vector<T> scratch_{};

private:
    [[nodiscard]] Derived const& derived_() const noexcept {
        return static_cast<Derived const&>(*this);
    }

    // Opt-in combine / scale, resolved at compile time. Absent → the site-action
    // defaults (raw neighbour sum, unit prefactor).
    [[nodiscard]] auto action_combine_() const noexcept {
        if constexpr (requires(Derived const& d) { d.action_combine(); }) {
            return derived_().action_combine();
        } else {
            return exec::IdentityCombine{};
        }
    }
    [[nodiscard]] auto force_combine_() const noexcept {
        if constexpr (requires(Derived const& d) { d.force_combine(); }) {
            return derived_().force_combine();
        } else {
            return exec::IdentityCombine{};
        }
    }
    [[nodiscard]] T action_scale_() const noexcept {
        if constexpr (requires(Derived const& d) { d.action_scale(); }) {
            return derived_().action_scale();
        } else {
            return T{1};
        }
    }
    void maybe_prep_(Lattice<T> const& l) const noexcept {
        if constexpr (requires(Derived const& d) { d.prep(l); }) {
            derived_().prep(l);
        }
    }
};

}  // namespace reticolo::action
