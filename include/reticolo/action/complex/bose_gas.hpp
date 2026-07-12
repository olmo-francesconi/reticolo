#pragma once

#include <reticolo/action/complex/complex_action.hpp>
#include <reticolo/action/complex/formula/bose_gas_formula.hpp>
#include <reticolo/action/complex/imag_part.hpp>
#include <reticolo/core/field/cplx.hpp>
#include <reticolo/core/field/field_traits.hpp>
#include <reticolo/core/field/lattice.hpp>
#include <reticolo/core/log/log.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>

namespace reticolo::action {

// std::complex <-> device-safe cplx<T> interop: the shared per-site formulas speak
// cplx<T>, while the CPU traversal hands the kernels std::complex. Inlines away.
template <class T>
[[gnu::always_inline]] inline cplx<T> to_cplx(std::complex<T> z) noexcept {
    return {z.real(), z.imag()};
}
template <class T>
[[gnu::always_inline]] inline std::complex<T> from_cplx(cplx<T> z) noexcept {
    return {z.re, z.im};
}

// Self-interacting relativistic lattice Bose gas at finite chemical potential.
// Two-component complex scalar phi_x on a d-dim hypercubic periodic lattice;
// the last direction is "time" and carries the chemical potential.
//
// Action (paper convention, arxiv:1910.11026 eq. 10):
//
//     S[phi] = sum_x [ (2d + m^2) |phi_x|^2  +  lambda |phi_x|^4
//                      - sum_{nu=1..d} (   phi*_x exp(-mu*delta_{nu,d}) phi_{x+nu}
//                                        + phi*_{x+nu} exp(+mu*delta_{nu,d}) phi_x ) ]
//
// Splitting phi = phi_1 + i phi_2 and unwrapping the hopping for the time
// direction yields S = S_R + i sinh(mu) S_I with
//
//     S_R = sum_x [ (2d + m^2) |phi_x|^2  +  lambda |phi_x|^4 ]
//         - 2 sum_x sum_{i=1..d-1} Re(phi*_x phi_{x+i_hat})
//         - 2 cosh(mu) sum_x       Re(phi*_x phi_{x+d_hat})
//
//     S_I = 2 sum_x  Im(phi*_x phi_{x+d_hat})
//
// Notes:
//  * S_R is what HMC samples (the phase-quenched ensemble). It depends on mu
//    via the cosh(mu) on the time-direction hopping.
//  * S_I is mu-independent — the chemical potential enters only as a Fourier
//    conjugate at reconstruction time (Z(mu) = integral rho(s) exp(-i sinh(mu) s) ds).
//  * Force convention: the integrator advances (phi_re, phi_im) as two
//    independent real DOFs, so compute_force writes the combined force
//    `F.re = -dS/dphi_re,  F.im = -dS/dphi_im` packaged as a single complex
//    per site. In Wirtinger notation that combination is F = -2 * dS/dphi*.
//  * At mu = 0 the action is O(2)-symmetric in (phi_1, phi_2).
//
// All the per-site arithmetic lives in `detail/bose_gas_formula.hpp` (shared
// with the CUDA device functors); the loop shells, the time-slab imag sweeps,
// the combined kick and the caches come from `ComplexAction`. This
// struct is the couplings + the four kernel binds + the cosh(mu) memo.

template <class T = double>
// NOLINTNEXTLINE(fuchsia-multiple-inheritance,misc-multiple-inheritance) — real base + imag mixin
struct BoseGas : ComplexAction<BoseGas<T>, T>, ImagPart<BoseGas<T>, T> {
    using value_type = std::complex<T>;
    using complex_t  = std::complex<T>;

    T mass   = T{1};
    T lambda = T{1};
    T mu     = T{0};

    void describe(log::Entry& e) const {
        e.line("BoseGas<{}>", scalar_name<value_type>());
        e.param("m={:.3f}", mass);
        e.param("λ={:.3f}", lambda);
        e.param("μ={:+.3f}", mu);
    }

    // S_R site: weighted forward sum (time dir ×cosh(mu)) → shared formula.
    [[nodiscard]] auto action_kernel(Lattice<complex_t> const& l) const noexcept {
        T const coef_mass  = (T{2} * static_cast<T>(l.ndims())) + (mass * mass);
        T const ch_minus_1 = cosh_mu_() - T{1};
        return [coef_mass, lam = lambda, ch_minus_1](
                   complex_t phi, complex_t fwd_total, complex_t fwd_last) {
            complex_t const weighted = fwd_total + (ch_minus_1 * fwd_last);
            return formula::bose_gas_action_site<T>(
                to_cplx(phi), to_cplx(weighted), coef_mass, lam);
        };
    }

    // F_R site: staple = all-neighbour sum with the time dir ×cosh(mu).
    [[nodiscard]] auto force_kernel(Lattice<complex_t> const& l) const noexcept {
        T const coef_mass  = (T{2} * static_cast<T>(l.ndims())) + (mass * mass);
        T const ch_minus_1 = cosh_mu_() - T{1};
        return [coef_mass, lam = lambda, ch_minus_1](
                   std::size_t /*i*/, complex_t phi, complex_t nbrs_total, complex_t nbrs_last) {
            complex_t const staple = nbrs_total + (ch_minus_1 * nbrs_last);
            cplx<T> const f =
                formula::bose_gas_force_site<T>(to_cplx(phi), to_cplx(staple), coef_mass, lam);
            return from_cplx(f);
        };
    }

    // S_I site = 2 Im(conj(phi_x)·phi_{x+tau}); the ×2 is inside the formula.
    [[nodiscard]] auto imag_action_kernel(Lattice<complex_t> const& /*l*/) const noexcept {
        return [](complex_t phi, complex_t phi_fwd_tau) {
            return formula::bose_gas_action_imag_site<T>(to_cplx(phi), to_cplx(phi_fwd_tau));
        };
    }

    // F_I site = 2i(phi_{x+tau} - phi_{x-tau}).
    [[nodiscard]] auto imag_force_kernel(Lattice<complex_t> const& /*l*/) const noexcept {
        return [](complex_t fwd_tau, complex_t bwd_tau) {
            cplx<T> const f =
                formula::bose_gas_force_imag_site<T>(to_cplx(fwd_tau), to_cplx(bwd_tau));
            return from_cplx(f);
        };
    }

    // cosh(mu) memo slots — public (like the caches) to keep the aggregate init.
    // NaN key != NaN so the first call always computes.
    mutable T cosh_mu_key_ = std::numeric_limits<T>::quiet_NaN();
    mutable T cosh_mu_val_ = std::numeric_limits<T>::quiet_NaN();

private:
    // cosh(mu) memoised on the current mu — a fixed action parameter, but the
    // kernels read it per build, so recomputing a cosh each time is pure overhead.
    [[nodiscard]] T cosh_mu_() const noexcept {
        if (mu != cosh_mu_key_) {
            cosh_mu_key_ = mu;
            cosh_mu_val_ = std::cosh(mu);
        }
        return cosh_mu_val_;
    }
};

}  // namespace reticolo::action
