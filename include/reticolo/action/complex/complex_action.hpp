#pragma once

#include <reticolo/action/cache.hpp>
#include <reticolo/action/sweep/complex.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/parallel.hpp>

#include <complex>
#include <cstddef>

// ComplexAction<Derived, T> — the common interface for complex scalar actions
// with a sign problem, i.e. a total S = S_R + i·S_I where HMC samples the
// real (phase-quenched) part S_R and S_I is the LLR constraint observable
// (BoseGas today). It provides the full `HasImagPart` surface on top of the
// baseline HMC one:
//
//   real part  : s_full / compute_force / compute_force_and_kick  (S_R, F_R)
//   imag part  : s_imag / compute_force_imag                      (S_I, F_I)
//   LLR mode B : compute_force_combined_and_kick                  (F_R + F_I)
//   caches     : last_s_full / last_s_imag (+ restore_*)
//
// The real part is anisotropic (the last direction — "time" — may carry a
// different weight, e.g. cosh(mu)), so it runs on the split-last drivers that
// hand the leaf the full neighbour sum *and* the last-direction sum separately.
// The imaginary part touches only the time direction, so it runs on the slab
// sweeps below. All physics is in the leaf's kernels, which receive the lattice
// (to read geometry-dependent prefactors like 2d + m²):
//
//   auto action_kernel(l) const;       // (phi, fwd_total, fwd_last) -> T (real S_R site)
//   auto force_kernel(l)  const;       // (i, phi, nbrs_total, nbrs_last) -> complex (F_R)
//   auto imag_action_kernel(l) const;  // (phi, phi_fwd_tau) -> T   (S_I site, ×2 folded in)
//   auto imag_force_kernel(l)  const;  // (phi_fwd_tau, phi_bwd_tau) -> complex (F_I)

namespace reticolo::action {

template <class Derived, class T>
struct ComplexAction : SFullCache, SImagCache {
    using complex_t = std::complex<T>;

    // ---- real part (phase-quenched, what HMC samples) ----

    [[nodiscard]] double s_full(Lattice<complex_t> const& l) const noexcept {
        auto kern             = derived_().action_kernel(l);
        complex_t const total = sweep::reduce_fwd_split_last<complex_t>(
            l, [&kern](complex_t phi, complex_t fwd_total, complex_t fwd_last) {
                return complex_t{kern(phi, fwd_total, fwd_last), T{0}};
            });
        double const s = static_cast<double>(std::real(total));
        last_s_full_   = s;
        return s;
    }

    void compute_force(Lattice<complex_t> const& l, Lattice<complex_t>& force) const noexcept {
        auto kern            = derived_().force_kernel(l);
        complex_t* const out = force.data();
        sweep::visit_nn_split_last<complex_t>(
            l, [&kern, out](std::size_t i, complex_t phi, complex_t nt, complex_t nl) {
                out[i] = kern(i, phi, nt, nl);
            });
    }

    void compute_force_and_kick(Lattice<complex_t> const& l,
                                Lattice<complex_t>& mom,
                                T k_dt) const noexcept {
        auto kern          = derived_().force_kernel(l);
        complex_t* const m = mom.data();
        sweep::visit_nn_split_last<complex_t>(
            l, [&kern, m, k_dt](std::size_t i, complex_t phi, complex_t nt, complex_t nl) {
                m[i] += k_dt * kern(i, phi, nt, nl);
            });
    }

    // ---- imaginary part (LLR constraint observable, time-direction only) ----

    [[nodiscard]] double s_imag(Lattice<complex_t> const& l) const noexcept {
        auto kern        = derived_().imag_action_kernel(l);
        double const acc = slab_reduce_tau_<double>(l, [&](complex_t self, complex_t fwd_tau) {
            return static_cast<double>(kern(self, fwd_tau));
        });
        last_s_imag_     = acc;
        return acc;
    }

    void compute_force_imag(Lattice<complex_t> const& l, Lattice<complex_t>& force) const noexcept {
        auto kern            = derived_().imag_force_kernel(l);
        complex_t* const out = force.data();
        slab_visit_tau_(l, [&](std::size_t i, complex_t fwd_tau, complex_t bwd_tau) {
            out[i] = kern(fwd_tau, bwd_tau);
        });
    }

    // Fused F_R + F_I into momentum: mom[i] += k_dt*(scale_r*F_R + scale_i*F_I),
    // avoiding the imag-force scratch buffer and the merge round-trip. Picked up
    // by WindowedAction (mode B) via concept detection.
    void compute_force_combined_and_kick(Lattice<complex_t> const& l,
                                         Lattice<complex_t>& mom,
                                         T scale_r,
                                         T scale_i,
                                         T k_dt) const noexcept {
        auto fk            = derived_().force_kernel(l);
        complex_t* const m = mom.data();
        T const k_r        = k_dt * scale_r;
        sweep::visit_nn_split_last<complex_t>(
            l, [&fk, m, k_r](std::size_t i, complex_t phi, complex_t nt, complex_t nl) {
                m[i] += k_r * fk(i, phi, nt, nl);
            });
        auto ik     = derived_().imag_force_kernel(l);
        T const k_i = k_dt * scale_i;
        slab_visit_tau_(l, [&](std::size_t i, complex_t fwd_tau, complex_t bwd_tau) {
            m[i] += k_i * ik(fwd_tau, bwd_tau);
        });
    }

private:
    [[nodiscard]] Derived const& derived_() const noexcept {
        return static_cast<Derived const&>(*this);
    }

    // Last dim τ ("time"): stride along τ is s_tau = nsites / L_tau, so the
    // (w,·) → (w±1,·) shift is a constant ±s_tau in the flat layout — a plain
    // slab sweep with no per-site neighbour-table lookup. Each τ-slab (fixed w) is
    // an independent work item: the visit is write-disjoint and the reduce folds
    // per-slab partials in canonical w order, so both are thread-count invariant.

    // Forward-τ reduction: body(self, phi_{x+τ}) -> Acc, summed over all sites.
    // Runs on the canonical field partition; every item lies inside ONE τ-slab
    // (the partition splits outer dims, τ outermost), so the ±τ shift is a
    // constant offset per item.
    template <class Acc, class Body>
    [[nodiscard]] Acc slab_reduce_tau_(Lattice<complex_t> const& l,
                                       Body const& body) const noexcept {
        std::size_t const d         = l.ndims();
        std::size_t const L_tau     = l.shape()[d - 1];
        std::size_t const s_tau     = l.nsites() / L_tau;
        std::size_t const wrap      = (L_tau - 1) * s_tau;  // top τ-slice → slice 0
        complex_t const* const data = l.data();
        // Per-site τ-forward neighbour (i+s_tau, wrapping the top slice to 0), so
        // the partition item may span any number of τ-slices — the slab shape is
        // free (a slab can now block the τ dim). No division: the wrap is a compare.
        return reticolo::exec::field_reduce<Acc>(
            l, 1, [&, data](std::size_t base, std::size_t cnt) {
                Acc partial{};
                for (std::size_t k = 0; k < cnt; ++k) {
                    std::size_t const i  = base + k;
                    std::size_t const ip = i >= wrap ? i - wrap : i + s_tau;
                    partial += body(data[i], data[ip]);
                }
                return partial;
            });
    }

    // Both-τ visit: body(i, phi_{x+τ}, phi_{x-τ}). Same canonical partition.
    template <class Body>
    void slab_visit_tau_(Lattice<complex_t> const& l, Body const& body) const noexcept {
        std::size_t const d         = l.ndims();
        std::size_t const L_tau     = l.shape()[d - 1];
        std::size_t const s_tau     = l.nsites() / L_tau;
        std::size_t const wrap      = (L_tau - 1) * s_tau;
        complex_t const* const data = l.data();
        // Per-site τ ± neighbours (wrap the top slice forward / slice 0 backward),
        // so an item may span any number of τ-slices — the slab shape is free.
        reticolo::exec::field_visit(l, 1, [&, data](std::size_t base, std::size_t cnt) {
            for (std::size_t k = 0; k < cnt; ++k) {
                std::size_t const i  = base + k;
                std::size_t const ip = i >= wrap ? i - wrap : i + s_tau;
                std::size_t const im = i < s_tau ? i + wrap : i - s_tau;
                body(i, data[ip], data[im]);
            }
        });
    }
};

}  // namespace reticolo::action
