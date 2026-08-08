#pragma once

#include <reticolo/action/cache.hpp>
#include <reticolo/action/concepts.hpp>
#include <reticolo/core/exec/nn_checkerboard.hpp>
#include <reticolo/core/exec/nn_split.hpp>
#include <reticolo/core/field/lattice.hpp>

#include <complex>
#include <cstddef>

// ComplexAction<Derived, T> — the REAL (phase-quenched) part of a complex scalar
// action with a sign problem, i.e. the S_R / F_R that HMC actually samples of a
// total S = S_R + i·S_I. It is an anisotropic nearest-neighbour action: the last
// direction ("time") may carry a different weight (e.g. cosh(mu)), so it runs on
// the split-last drivers that hand the leaf the full neighbour sum AND the
// last-direction sum separately. All physics is in the leaf's kernels, which
// receive the lattice (to read geometry-dependent prefactors like 2d + m²):
//
//   auto action_kernel(l) const;  // (phi, fwd_total, fwd_last) -> T       (S_R site)
//   auto force_kernel(l)  const;  // (i, phi, nbrs_total, nbrs_last) -> complex (F_R)
//
// The IMAGINARY part (S_I, F_I — the LLR constraint observable) is an orthogonal
// extension: the leaf derives from `ImagPart` (action/complex/imag_part.hpp)
// ALONGSIDE this base, mirroring how WindowedAction decorates a base action. So a
// complex leaf is `struct BoseGas : ComplexAction<…>, ImagPart<…>`.

namespace reticolo::action {

template <class Derived, class T>
struct ComplexAction : SFullCache {
    using complex_t = std::complex<T>;

    [[nodiscard]] double s_full(Lattice<complex_t> const& l) const noexcept {
        auto kern             = derived_().action_kernel(l);
        complex_t const total = exec::nn_reduce_fwd_split_last<complex_t>(
            l, [&kern](complex_t phi, complex_t fwd_total, complex_t fwd_last) {
                return complex_t{kern(phi, fwd_total, fwd_last), T{0}};
            });
        auto const s = static_cast<double>(std::real(total));
        last_s_full_ = s;
        return s;
    }

    void compute_force(Lattice<complex_t> const& l, Lattice<complex_t>& force) const noexcept {
        auto kern            = derived_().force_kernel(l);
        complex_t* const out = force.data();
        exec::nn_visit_split_last<complex_t>(
            l, [&kern, out](std::size_t i, complex_t phi, complex_t nt, complex_t nl) {
                out[i] = kern(i, phi, nt, nl);
            });
    }

    void compute_force_and_kick(Lattice<complex_t> const& l,
                                Lattice<complex_t>& mom,
                                T k_dt) const noexcept {
        auto kern          = derived_().force_kernel(l);
        complex_t* const m = mom.data();
        exec::nn_visit_split_last<complex_t>(
            l, [&kern, m, k_dt](std::size_t i, complex_t phi, complex_t nt, complex_t nl) {
                m[i] += k_dt * kern(i, phi, nt, nl);
            });
    }

    // One colour of a local (Metropolis) sweep, on the PHASE-QUENCHED weight —
    // the same construction as NNAction's, and free from the leaf's own
    // `action_kernel` for the same reason: the per-site S_R of x is exactly
    // K(φ_x, total, last) once the aggregates run over ALL 2d neighbours instead
    // of the d forward ones.
    //
    // That the identity survives the anisotropy is worth stating, because it is
    // where a complex action could have broken it. The hopping term is
    // −2·Re(conj(φ_x)·weighted_nbr), and Re(conj(a)·b) is SYMMETRIC in a and b —
    // so a bond contributes the same to either endpoint's local energy and the
    // "each touching bond once" counting holds. The chemical potential's
    // forward/backward asymmetry — the actual source of the sign problem — lives
    // entirely in `s_imag`, which the phase-quenched sweep never evaluates.
    //
    // Proposal is φ' = φ + σ·noise with `noise` an independent complex Gaussian
    // (fill_gaussian gives independent N(0,1) on Re and Im), so a move explores
    // both components at once rather than needing two sub-sweeps.
    [[nodiscard]] LocalStats metropolis_stencil(Lattice<complex_t>& l,
                                                int color,
                                                Lattice<complex_t> const& noise,
                                                Lattice<T> const& logu,
                                                T sigma) const noexcept {
        auto const kern            = derived_().action_kernel(l);
        std::size_t const last_dim = l.ndims() - 1;
        complex_t* const data      = l.data();
        complex_t const* const nz  = noise.data();
        T const* const lu          = logu.data();

        return exec::nn_color_reduce<LocalStats>(
            l, color, [&](std::size_t i, complex_t phi, auto const& gather) {
                // The aggregates do NOT depend on the candidate here (the combine
                // is the identity — the anisotropy weight is applied by the leaf
                // to the sums, not per bond), so ONE gather serves both energies.
                complex_t total{};
                complex_t last{};
                gather([&](std::size_t mu, complex_t nbr) {
                    total += nbr;
                    if (mu == last_dim) {
                        last += nbr;
                    }
                });
                complex_t const prop = phi + (sigma * nz[i]);
                auto const ds        = static_cast<double>(kern(prop, total, last)) -
                                       static_cast<double>(kern(phi, total, last));
                if (-ds >= static_cast<double>(lu[i])) {
                    data[i] = prop;
                    return LocalStats{.n_accepted = 1, .n_attempts = 1, .ds = ds};
                }
                return LocalStats{.n_accepted = 0, .n_attempts = 1, .ds = 0.0};
            });
    }

    // Two checkerboard colours — a site field's elementary move is per site.
    [[nodiscard]] static constexpr int n_colors(Lattice<complex_t> const& /*l*/) noexcept {
        return 2;
    }

private:
    [[nodiscard]] Derived const& derived_() const noexcept {
        return static_cast<Derived const&>(*this);
    }
};

}  // namespace reticolo::action
