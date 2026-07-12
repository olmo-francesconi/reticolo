#pragma once

#include <reticolo/action/cache.hpp>
#include <reticolo/action/sweep/complex.hpp>
#include <reticolo/core/field/lattice.hpp>
#include <reticolo/core/exec/parallel.hpp>

#include <complex>
#include <cstddef>

// ImagPart<Derived, T> — the imaginary-part EXTENSION for complex scalar actions
// with a sign problem. A stateless mixin the leaf derives from ALONGSIDE its
// real-part base (ComplexAction): the real base carries s_full / compute_force
// (S_R, F_R — the phase-quenched part HMC samples), and this mixin adds the
// `HasImagPart` surface — s_imag / compute_force_imag (S_I, F_I, the LLR
// constraint observable, which touches only the time direction) plus the fused
// combined kick. It is the decorator analogue of `WindowedAction`: a small
// capability bolted onto a real action rather than a whole separate family. The
// leaf supplies two more kernels:
//
//   auto imag_action_kernel(l) const;  // (phi, phi_fwd_tau) -> T   (S_I site, ×2 folded)
//   auto imag_force_kernel(l)  const;  // (phi_fwd_tau, phi_bwd_tau) -> complex (F_I)
//
// The imaginary observable depends only on the time direction τ (dim D-1), so it
// runs on the τ-slab drivers below rather than the full neighbour stencil.

namespace reticolo::action {

template <class Derived, class T>
struct ImagPart : SImagCache {
    using complex_t = std::complex<T>;

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

    // Fused F_R + F_I into momentum: mom[i] += k_dt·(scale_r·F_R + scale_i·F_I),
    // avoiding the imag-force scratch buffer and the merge round-trip. Picked up
    // by WindowedAction (mode B) via concept detection. Uses the real force kernel
    // from the derived leaf (split-last, anisotropic) for the F_R sweep.
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
    // (w,·) → (w±1,·) shift is a constant ±s_tau in the flat layout — a plain slab
    // sweep with no per-site neighbour-table lookup. Each τ-slab (fixed w) is an
    // independent work item: the visit is write-disjoint and the reduce folds
    // per-slab partials in canonical w order, so both are thread-count invariant.

    // Forward-τ reduction: body(self, phi_{x+τ}) -> Acc, summed over all sites.
    template <class Acc, class Body>
    [[nodiscard]] Acc slab_reduce_tau_(Lattice<complex_t> const& l,
                                       Body const& body) const noexcept {
        std::size_t const d         = l.ndims();
        std::size_t const l_tau     = l.shape()[d - 1];
        std::size_t const s_tau     = l.nsites() / l_tau;
        std::size_t const wrap      = (l_tau - 1) * s_tau;  // top τ-slice → slice 0
        complex_t const* const data = l.data();
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
        std::size_t const l_tau     = l.shape()[d - 1];
        std::size_t const s_tau     = l.nsites() / l_tau;
        std::size_t const wrap      = (l_tau - 1) * s_tau;
        complex_t const* const data = l.data();
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
