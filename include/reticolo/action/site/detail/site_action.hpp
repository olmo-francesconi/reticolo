#pragma once

#include <reticolo/action/site/detail/traversal.hpp>
#include <reticolo/core/lattice.hpp>

#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

// SiteAction<Derived, T> — the common interface for nearest-neighbour scalar
// actions on a `Lattice<T>` (Phi4, Phi6, SineGordon, …). It owns everything a
// site action shares: the `reduce_fwd` / `visit_nn` loop shells, the fused
// force+kick pass, the `last_s_full` cache, and the LLR `s_full_and_force`
// fast-path. The leaf action is a small aggregate that derives from this base
// and supplies only the physics, as coupling-hoisting kernels binding the
// shared `detail/*_formula.hpp` functions:
//
//   auto force_kernel()  const;   // (size_t i, T phi, T nbrs) -> T   (-dS/dphi)
//   auto action_kernel() const;   // (T phi, T fwd) -> T              (S(x))
//
// Both return a callable that captures the couplings by value, so the base
// hoists them into the hot loop exactly as a hand-written action would — the
// vectorised inner loop is unchanged. The kernels take no member access per
// site.
//
// Optional refinements the base picks up by concept when the leaf provides them:
//   void prep(Lattice<T> const&) const;         // fill scratch() before a force pass
//                                               // (e.g. SineGordon's sin/cos batch)
//   auto force_energy_kernel() const;           // (size_t i, T phi, T nbrs)
//                                               //   -> pair<T force, T s_site>
//                                               // enables s_full_and_force (LLR)
//
// Leaves whose s_full does not fit the plain `action_kernel` form (SineGordon's
// batched-transcendental total) simply declare their own `s_full`, which hides
// the base one; the shared cache still lives here.

namespace reticolo::action::detail {

template <class Derived, class T>
struct SiteAction {
    [[nodiscard]] double s_full(Lattice<T> const& l) const noexcept {
        auto kern      = derived_().action_kernel();
        double const s = reduce_fwd<T, double>(
            l, [&kern](T phi, T fwd) { return static_cast<double>(kern(phi, fwd)); });
        last_s_full_ = s;
        return s;
    }

    void compute_force(Lattice<T> const& l, Lattice<T>& force) const noexcept {
        maybe_prep_(l);
        auto kern    = derived_().force_kernel();
        T* const out = force.data();
        // visit_nn self-threads via the parallel primitives (write-disjoint map).
        visit_nn<T>(l, [&kern, out](std::size_t i, T phi, T nbrs) { out[i] = kern(i, phi, nbrs); });
    }

    void compute_force_and_kick(Lattice<T> const& l, Lattice<T>& mom, T k_dt) const noexcept {
        maybe_prep_(l);
        auto kern  = derived_().force_kernel();
        T* const m = mom.data();
        visit_nn<T>(l, [&kern, m, k_dt](std::size_t i, T phi, T nbrs) {
            m[i] += k_dt * kern(i, phi, nbrs);
        });
    }

    // Fused total action + force in one neighbour pass, driven by a leaf's
    // force-energy kernel `(i, phi, nbrs) -> {force, s_site}`. Per-site action
    // values are staged in scratch and reduced in a separate linear pass:
    // accumulating into a scalar inside the visit body would put a loop-carried
    // FP dependency in the force loop and serialise it. Does not touch the
    // `last_s_full` cache (that stays with `s_full`).
    //
    // Exposed as the public LLR entry point `s_full_and_force` only by leaves
    // that opt in (Phi4) — kept out of the base itself so the WindowedAction's
    // `requires { base.s_full_and_force(...) }` probe stays a clean present/absent
    // test rather than an inherited-but-unsatisfied hard error.
    template <class Kernel>
    [[nodiscard]] double
    staged_force_energy(Lattice<T> const& l, Lattice<T>& force, Kernel&& kern) const noexcept {
        T* const out        = force.data();
        std::size_t const n = l.nsites();
        ensure_scratch(n);
        T* const sb = scratch_.data();
        // Force + staged per-site action: write-disjoint map (self-threaded).
        visit_nn<T>(l, [&kern, out, sb](std::size_t i, T phi, T nbrs) {
            auto const [f, s] = kern(i, phi, nbrs);
            out[i]            = f;
            sb[i]             = s;
        });
        // Reduce the staged action in a fixed-block partition (thread-count
        // invariant), matching s_full's parallel reduce.
        return reticolo::detail::parallel_reduce_ranges(reticolo::detail::traverse_want(n),
                                                        n,
                                                        k_site_chunk,
                                                        [sb](std::size_t base, std::size_t cnt) {
                                                            RETICOLO_FP_REASSOCIATE
                                                            double s              = 0.0;
                                                            std::size_t const end = base + cnt;
                                                            for (std::size_t i = base; i < end;
                                                                 ++i) {
                                                                s += static_cast<double>(sb[i]);
                                                            }
                                                            return s;
                                                        });
    }

    [[nodiscard]] double last_s_full() const noexcept { return last_s_full_; }
    void restore_last_s_full(double v) const noexcept { last_s_full_ = v; }

    // Lazy per-site scratch, sized to nsites — shared by the LLR fast-path and by
    // any leaf `prep`/`s_full` that batches a transcendental (SineGordon). Public
    // (like the caches) to keep the derived struct an aggregate.
    void ensure_scratch(std::size_t n) const {
        if (scratch_.size() < n) {
            scratch_.resize(n);
        }
    }

    mutable std::vector<T> scratch_{};
    mutable double last_s_full_ = std::numeric_limits<double>::quiet_NaN();

private:
    [[nodiscard]] Derived const& derived_() const noexcept {
        return static_cast<Derived const&>(*this);
    }

    void maybe_prep_(Lattice<T> const& l) const noexcept {
        if constexpr (requires(Derived const& d) { d.prep(l); }) {
            derived_().prep(l);
        }
    }
};

}  // namespace reticolo::action::detail
