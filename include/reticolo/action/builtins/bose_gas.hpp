#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/site.hpp>

#include <cmath>
#include <complex>
#include <cstddef>

namespace reticolo::action {

// =============================================================================
//  Self-interacting relativistic lattice Bose gas at finite chemical potential.
//  Two-component complex scalar phi_x on a d-dim hypercubic periodic lattice;
//  the last direction is "time" and carries the chemical potential.
//
//  Action (paper convention, arxiv:1910.11026 eq. 10):
//
//      S[phi] = sum_x [ (2d + m^2) |phi_x|^2  +  lambda |phi_x|^4
//                       - sum_{nu=1..d} (   phi*_x exp(-mu*delta_{nu,d}) phi_{x+nu}
//                                         + phi*_{x+nu} exp(+mu*delta_{nu,d}) phi_x ) ]
//
//  Splitting phi = phi_1 + i phi_2 and unwrapping the hopping for the time
//  direction yields S = S_R + i sinh(mu) S_I with
//
//      S_R = sum_x [ (2d + m^2) |phi_x|^2  +  lambda |phi_x|^4 ]
//          - 2 sum_x sum_{i=1..d-1} Re(phi*_x phi_{x+i_hat})
//          - 2 cosh(mu) sum_x       Re(phi*_x phi_{x+d_hat})
//
//      S_I = 2 sum_x  Im(phi*_x phi_{x+d_hat})
//
//  Notes:
//   * S_R is what HMC samples (the phase-quenched ensemble). It depends on mu
//     via the cosh(mu) on the time-direction hopping.
//   * S_I is mu-independent — the chemical potential enters only as a Fourier
//     conjugate at reconstruction time (Z(mu) = integral rho(s) exp(-i sinh(mu) s) ds).
//   * Force convention: the integrator advances (phi_re, phi_im) as two
//     independent real DOFs, so compute_force writes the combined force
//     `F.re = -dS/dphi_re,  F.im = -dS/dphi_im` packaged as a single complex
//     per site. In Wirtinger notation that combination is F = -2 * dS/dphi*,
//     so all closed-form expressions below carry the factor of 2.
//   * At mu = 0 the action is O(2)-symmetric in (phi_1, phi_2) and reduces to
//     a complex-field rewriting of the standard relativistic Bose gas.
// =============================================================================

template <class T = double>
struct BoseGas {
    using value_type = std::complex<T>;
    using complex_t  = std::complex<T>;

    T mass   = T{1};
    T lambda = T{1};
    T mu     = T{0};

    // ---------- LocalAction (Metropolis would need these) ------------------

    [[nodiscard]] T s_local(Lattice<complex_t> const& l, Site x) const noexcept {
        std::size_t const d = l.ndims();
        T const coef_mass   = (T{2} * static_cast<T>(d)) + (mass * mass);
        T const ch          = std::cosh(mu);
        complex_t const phi = l[x];
        T const abs2        = std::norm(phi);
        complex_t staple{T{0}, T{0}};
        for (std::size_t nu = 0; nu < d; ++nu) {
            T const c = (nu + 1 == d) ? ch : T{1};
            staple += c * (l[l.next(x, nu)] + l[l.prev(x, nu)]);
        }
        return (coef_mass * abs2) + (lambda * abs2 * abs2) -
               (T{2} * std::real(std::conj(phi) * staple));
    }

    [[nodiscard]] T ds_local(Lattice<complex_t> const& l, Site x, complex_t new_v) const noexcept {
        std::size_t const d = l.ndims();
        T const coef_mass   = (T{2} * static_cast<T>(d)) + (mass * mass);
        T const ch          = std::cosh(mu);
        complex_t const phi = l[x];
        T const a_old       = std::norm(phi);
        T const a_new       = std::norm(new_v);
        complex_t staple{T{0}, T{0}};
        for (std::size_t nu = 0; nu < d; ++nu) {
            T const c = (nu + 1 == d) ? ch : T{1};
            staple += c * (l[l.next(x, nu)] + l[l.prev(x, nu)]);
        }
        return (coef_mass * (a_new - a_old)) + (lambda * ((a_new * a_new) - (a_old * a_old))) -
               (T{2} * std::real(std::conj(new_v - phi) * staple));
    }

    // ---------- HasSEff: real (phase-quenched) part ------------------------

    [[nodiscard]] T s_full(Lattice<complex_t> const& l) const noexcept {
        std::size_t const d = l.ndims();
        T const coef_mass   = (T{2} * static_cast<T>(d)) + (mass * mass);
        T const ch          = std::cosh(mu);
        T onsite            = T{0};
        T hopping           = T{0};
        for (Site x : l.sites()) {
            T const abs2 = std::norm(l[x]);
            onsite += (coef_mass * abs2) + (lambda * abs2 * abs2);
            for (std::size_t nu = 0; nu < d; ++nu) {
                T const c         = (nu + 1 == d) ? ch : T{1};
                complex_t const v = std::conj(l[x]) * l[l.next(x, nu)];
                hopping += c * std::real(v);
            }
        }
        return onsite - (T{2} * hopping);
    }

    // ---------- HasForce: F_R = -dS_R/dphi* --------------------------------

    void compute_force(Lattice<complex_t> const& l, Lattice<complex_t>& force) const noexcept {
        std::size_t const d = l.ndims();
        T const coef_mass   = (T{2} * static_cast<T>(d)) + (mass * mass);
        T const ch          = std::cosh(mu);
        for (Site x : l.sites()) {
            complex_t const phi = l[x];
            T const abs2        = std::norm(phi);
            complex_t staple{T{0}, T{0}};
            for (std::size_t nu = 0; nu < d; ++nu) {
                T const c = (nu + 1 == d) ? ch : T{1};
                staple += c * (l[l.next(x, nu)] + l[l.prev(x, nu)]);
            }
            // F_R.re = -dS_R/dphi_re, F_R.im = -dS_R/dphi_im. Combined as a
            // complex this is -2 * dS_R/dphi*; on-site + hopping with the
            // explicit factor of 2:
            //   F_R(x) = -2(2d+m^2) phi_x - 4 lambda |phi|^2 phi_x + 2 staple
            force[x] = (T{-2} * coef_mass * phi) - (T{4} * lambda * abs2 * phi) + (T{2} * staple);
        }
    }

    // ---------- HasImagPart: S_I and F_I = -dS_I/dphi* ---------------------

    // Local change in S_I when phi_x → new_v. Only the two time-direction
    // hopping terms touching x contribute. Used by the windowed Metropolis
    // cascade-thermalization in `apps/bose_gas_llr.cpp` so the system can be
    // pulled into each LLR window even at force scales where HMC would lose
    // integrator stability.
    [[nodiscard]] T
    ds_imag_local(Lattice<complex_t> const& l, Site x, complex_t new_v) const noexcept {
        std::size_t const tau   = l.ndims() - 1;
        complex_t const phi_x   = l[x];
        complex_t const phi_pmu = l[l.next(x, tau)];
        complex_t const phi_mmu = l[l.prev(x, tau)];
        T const old_t =
            T{2} * (std::imag(std::conj(phi_x) * phi_pmu) + std::imag(std::conj(phi_mmu) * phi_x));
        T const new_t =
            T{2} * (std::imag(std::conj(new_v) * phi_pmu) + std::imag(std::conj(phi_mmu) * new_v));
        return new_t - old_t;
    }

    [[nodiscard]] T s_imag(Lattice<complex_t> const& l) const noexcept {
        std::size_t const d   = l.ndims();
        std::size_t const tau = d - 1;
        T acc                 = T{0};
        for (Site x : l.sites()) {
            complex_t const v = std::conj(l[x]) * l[l.next(x, tau)];
            acc += std::imag(v);
        }
        return T{2} * acc;
    }

    void compute_force_imag(Lattice<complex_t> const& l, Lattice<complex_t>& force) const noexcept {
        // S_I = 2 sum_x Im(phi*_x phi_{x+d_hat})
        //     = -i sum_x ( phi*_x phi_{x+d_hat} - phi_x phi*_{x+d_hat} )
        // dS_I/dphi*(x) = -i ( phi_{x+d_hat} - phi_{x-d_hat} )
        // F_I(x) = -2 * dS_I/dphi*(x) = 2i ( phi_{x+d_hat} - phi_{x-d_hat} )
        std::size_t const tau = l.ndims() - 1;
        complex_t const two_i{T{0}, T{2}};
        for (Site x : l.sites()) {
            force[x] = two_i * (l[l.next(x, tau)] - l[l.prev(x, tau)]);
        }
    }
};

}  // namespace reticolo::action
