#pragma once

#include <reticolo/action/detail/helpers.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/indexing.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/site.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>

namespace reticolo::action {

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
//    per site. In Wirtinger notation that combination is F = -2 * dS/dphi*,
//    so all closed-form expressions below carry the factor of 2.
//  * At mu = 0 the action is O(2)-symmetric in (phi_1, phi_2) and reduces to
//    a complex-field rewriting of the standard relativistic Bose gas.

template <class T = double>
struct BoseGas {
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

    [[nodiscard]] T s_local(Lattice<complex_t> const& l, Site x) const noexcept {
        std::size_t const d = l.ndims();
        T const coef_mass   = (T{2} * static_cast<T>(d)) + (mass * mass);
        T const ch          = cosh_mu_();
        complex_t const phi = l[x];
        T const abs2        = std::norm(phi);
        complex_t staple{T{0}, T{0}};
        for (std::size_t nu = 0; nu < d; ++nu) {
            T const c = (nu + 1 == d) ? ch : T{1};
            staple += c * (l[l.next(x, nu)] + l[l.prev(x, nu)]);
        }
        return (coef_mass * abs2) + (lambda * abs2 * abs2) - (T{2} * re_conj_mul_(phi, staple));
    }

    [[nodiscard]] T ds_local(Lattice<complex_t> const& l, Site x, complex_t new_v) const noexcept {
        std::size_t const d = l.ndims();
        T const coef_mass   = (T{2} * static_cast<T>(d)) + (mass * mass);
        T const ch          = cosh_mu_();
        complex_t const phi = l[x];
        T const a_old       = std::norm(phi);
        T const a_new       = std::norm(new_v);
        complex_t staple{T{0}, T{0}};
        for (std::size_t nu = 0; nu < d; ++nu) {
            T const c = (nu + 1 == d) ? ch : T{1};
            staple += c * (l[l.next(x, nu)] + l[l.prev(x, nu)]);
        }
        return (coef_mass * (a_new - a_old)) + (lambda * ((a_new * a_new) - (a_old * a_old))) -
               (T{2} * re_conj_mul_(new_v - phi, staple));
    }

    [[nodiscard]] T s_full(Lattice<complex_t> const& l) const noexcept {
        std::size_t const d   = l.ndims();
        T const coef_mass     = (T{2} * static_cast<T>(d)) + (mass * mass);
        T const ch_minus_1    = cosh_mu_() - T{1};
        complex_t const total = detail::reduce_fwd_split_last<complex_t>(
            l,
            [coef_mass, lam = lambda, ch_minus_1](
                complex_t phi, complex_t fwd_total, complex_t fwd_last) {
                T const abs2 = std::norm(phi);
                // hopping(x) = real( conj(phi) * (fwd_total + (ch-1)*fwd_last) )
                complex_t const weighted = fwd_total + (ch_minus_1 * fwd_last);
                T const hop              = re_conj_mul_(phi, weighted);
                return complex_t{(coef_mass * abs2) + (lam * abs2 * abs2) - (T{2} * hop), T{0}};
            });
        T const s    = std::real(total);
        last_s_full_ = s;
        return s;
    }

    [[nodiscard]] T last_s_full() const noexcept { return last_s_full_; }
    void restore_last_s_full(T v) const noexcept { last_s_full_ = v; }

    void compute_force(Lattice<complex_t> const& l, Lattice<complex_t>& force) const noexcept {
        std::size_t const d  = l.ndims();
        T const coef_mass    = (T{2} * static_cast<T>(d)) + (mass * mass);
        T const ch_minus_1   = cosh_mu_() - T{1};
        complex_t* const out = force.data();
        // F_R(x) = -2(2d+m^2) phi_x - 4 lambda |phi|^2 phi_x + 2 * staple
        // staple = sum_{mu, +-} phi_{x+mu} + (ch-1) * (phi_{x+last} + phi_{x-last})
        //        = nbrs_total + (ch-1) * nbrs_last
        detail::visit_nn_split_last<complex_t>(
            l,
            [coef_mass, lam = lambda, ch_minus_1, out](
                std::size_t i, complex_t phi, complex_t nbrs_total, complex_t nbrs_last) {
                T const abs2           = std::norm(phi);
                complex_t const staple = nbrs_total + (ch_minus_1 * nbrs_last);
                out[i] = (T{-2} * coef_mass * phi) - (T{4} * lam * abs2 * phi) + (T{2} * staple);
            });
    }

    // Fused force-and-kick: mom += k_dt * F_R[i] in one visit_nn pass.
    // Satisfies HasFusedKick, which lets WindowedAction expose its own
    // compute_force_and_kick and gives the LLR complex path access to the
    // combined-kick kernel below.
    void compute_force_and_kick(Lattice<complex_t> const& l,
                                Lattice<complex_t>& mom,
                                complex_t k_dt) const noexcept {
        std::size_t const d = l.ndims();
        T const coef_mass   = (T{2} * static_cast<T>(d)) + (mass * mass);
        T const ch_minus_1  = cosh_mu_() - T{1};
        complex_t* const mp = mom.data();
        detail::visit_nn_split_last<complex_t>(
            l,
            [coef_mass, lam = lambda, ch_minus_1, mp, k_dt](
                std::size_t i, complex_t phi, complex_t nbrs_total, complex_t nbrs_last) {
                T const abs2           = std::norm(phi);
                complex_t const staple = nbrs_total + (ch_minus_1 * nbrs_last);
                complex_t const f_r =
                    (T{-2} * coef_mass * phi) - (T{4} * lam * abs2 * phi) + (T{2} * staple);
                mp[i] += k_dt * f_r;
            });
    }

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
        T const old_t = T{2} * (im_conj_mul_(phi_x, phi_pmu) + im_conj_mul_(phi_mmu, phi_x));
        T const new_t = T{2} * (im_conj_mul_(new_v, phi_pmu) + im_conj_mul_(phi_mmu, new_v));
        return new_t - old_t;
    }

    [[nodiscard]] T s_imag(Lattice<complex_t> const& l) const noexcept {
        // S_I = 2 sum_x Im(phi*_x phi_{x+tau_hat}).
        // Last dim is "time" tau. Stride along tau is s_tau = nsites / L_tau,
        // so the (w_tau, ...) → (w_tau+1, ...) shift is a constant +s_tau in
        // the flat layout — bulk-vs-slab pattern works directly without a
        // per-site indexing table lookup.
        std::size_t const d       = l.ndims();
        std::size_t const L_tau   = l.shape()[d - 1];
        std::size_t const n       = l.nsites();
        std::size_t const s_tau   = n / L_tau;
        complex_t const* const in = l.data();
        T acc                     = T{0};
        for (std::size_t w = 0; w < L_tau; ++w) {
            std::size_t const wp     = (w + 1 == L_tau) ? 0 : (w + 1);
            std::size_t const base   = w * s_tau;
            std::size_t const base_p = wp * s_tau;
            for (std::size_t k = 0; k < s_tau; ++k) {
                acc += im_conj_mul_(in[base + k], in[base_p + k]);
            }
        }
        T const s    = T{2} * acc;
        last_s_imag_ = s;
        return s;
    }

    [[nodiscard]] T last_s_imag() const noexcept { return last_s_imag_; }
    void restore_last_s_imag(T v) const noexcept { last_s_imag_ = v; }

    void compute_force_imag(Lattice<complex_t> const& l, Lattice<complex_t>& force) const noexcept {
        // F_I(x) = 2i ( phi_{x+tau_hat} - phi_{x-tau_hat} ) — only the
        // time-direction neighbours contribute, same slab pattern as s_imag.
        std::size_t const d       = l.ndims();
        std::size_t const L_tau   = l.shape()[d - 1];
        std::size_t const n       = l.nsites();
        std::size_t const s_tau   = n / L_tau;
        complex_t const* const in = l.data();
        complex_t* const out      = force.data();
        complex_t const two_i{T{0}, T{2}};
        for (std::size_t w = 0; w < L_tau; ++w) {
            std::size_t const wm     = (w == 0) ? (L_tau - 1) : (w - 1);
            std::size_t const wp     = (w + 1 == L_tau) ? 0 : (w + 1);
            std::size_t const base   = w * s_tau;
            std::size_t const base_m = wm * s_tau;
            std::size_t const base_p = wp * s_tau;
            for (std::size_t k = 0; k < s_tau; ++k) {
                out[base + k] = two_i * (in[base_p + k] - in[base_m + k]);
            }
        }
    }

    //
    //   mom[i] += k_dt * ( scale_r * F_R[i] + scale_i * F_I[i] )
    //
    // Equivalent to:
    //   compute_force_and_kick(l, mom, k_dt * scale_r);
    //   compute_force_imag(l, imag_scratch);
    //   for (i) mom[i] += k_dt * scale_i * imag_scratch[i];
    //
    // but avoids the imag_scratch allocation and the merge-pass round trip.
    // `WindowedAction` picks this up via concept-detection in mode B and uses
    // it when present (BoseGas today, future complex-action implementations
    // tomorrow).
    void compute_force_combined_and_kick(Lattice<complex_t> const& l,
                                         Lattice<complex_t>& mom,
                                         T scale_r,
                                         T scale_i,
                                         T k_dt) const noexcept {
        std::size_t const d = l.ndims();
        T const coef_mass   = (T{2} * static_cast<T>(d)) + (mass * mass);
        T const ch_minus_1  = cosh_mu_() - T{1};
        complex_t* const mp = mom.data();
        T const k_r         = k_dt * scale_r;

        // F_R contribution — same generic visit as compute_force, but
        // accumulates `k_r * F_R[i]` directly into mom instead of writing
        // to a separate buffer.
        detail::visit_nn_split_last<complex_t>(
            l,
            [coef_mass, lam = lambda, ch_minus_1, mp, k_r](
                std::size_t i, complex_t phi, complex_t nbrs_total, complex_t nbrs_last) {
                T const abs2           = std::norm(phi);
                complex_t const staple = nbrs_total + (ch_minus_1 * nbrs_last);
                complex_t const f_r =
                    (T{-2} * coef_mass * phi) - (T{4} * lam * abs2 * phi) + (T{2} * staple);
                mp[i] += k_r * f_r;
            });

        // F_I contribution — time-only sweep, accumulates directly into mom.
        std::size_t const L_tau   = l.shape()[d - 1];
        std::size_t const n       = l.nsites();
        std::size_t const s_tau   = n / L_tau;
        complex_t const* const in = l.data();
        T const k_i               = k_dt * scale_i;
        complex_t const k_i_two_i{T{0}, T{2} * k_i};  // pre-scale 2i by k_i
        for (std::size_t w = 0; w < L_tau; ++w) {
            std::size_t const wm     = (w == 0) ? (L_tau - 1) : (w - 1);
            std::size_t const wp     = (w + 1 == L_tau) ? 0 : (w + 1);
            std::size_t const base   = w * s_tau;
            std::size_t const base_m = wm * s_tau;
            std::size_t const base_p = wp * s_tau;
            for (std::size_t k = 0; k < s_tau; ++k) {
                mp[base + k] += k_i_two_i * (in[base_p + k] - in[base_m + k]);
            }
        }
    }

    mutable T last_s_full_ = std::numeric_limits<T>::quiet_NaN();
    mutable T last_s_imag_ = std::numeric_limits<T>::quiet_NaN();

    // cosh(mu) memo slots — public (like the caches above) to preserve
    // aggregate-init; read only through `cosh_mu_()`. NaN key != NaN so the
    // first call always computes.
    mutable T cosh_mu_key_ = std::numeric_limits<T>::quiet_NaN();
    mutable T cosh_mu_val_ = std::numeric_limits<T>::quiet_NaN();

private:
    // Re(conj(a) * b) and Im(conj(a) * b) without the full complex product —
    // 2 mul + 1 add instead of 4 mul + 2 add with half the result discarded.
    [[nodiscard]] static T re_conj_mul_(complex_t a, complex_t b) noexcept {
        return (a.real() * b.real()) + (a.imag() * b.imag());
    }
    [[nodiscard]] static T im_conj_mul_(complex_t a, complex_t b) noexcept {
        return (a.real() * b.imag()) - (a.imag() * b.real());
    }

    // cosh(mu) memoised on the current mu — `mu` is a fixed action parameter,
    // but s_local/ds_local run per proposed site on the Metropolis cascade
    // path where a fresh cosh per call is pure overhead.
    [[nodiscard]] T cosh_mu_() const noexcept {
        if (mu != cosh_mu_key_) {
            cosh_mu_key_ = mu;
            cosh_mu_val_ = std::cosh(mu);
        }
        return cosh_mu_val_;
    }
};

}  // namespace reticolo::action
