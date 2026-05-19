#pragma once

#include <reticolo/action/detail/gauge_group/su2.hpp>
#include <reticolo/action/detail/gauge_group/su3.hpp>
#include <reticolo/action/detail/gauge_group/u1.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/math/su2_ops.hpp>
#include <reticolo/math/su3_ops.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace reticolo::bench {

// =============================================================================
//  Hot-start helpers — fill each field type with a non-trivial random
//  configuration so kernel benches measure real arithmetic instead of
//  zero-fold shortcuts. Init time is excluded from the timed loop by
//  construction (callers run this *before* the timing harness).
//
//  Each overload follows the same pattern: take a freshly-default-
//  constructed field, fill in place from an Rng. The Rng's state is
//  advanced; reuse a single rng across multiple init calls if you want
//  decorrelated fields.
// =============================================================================

// ---- Lattice<double> ---------------------------------------------------------
template <class Rng>
void hot_init(Lattice<double>& f, Rng& rng) noexcept {
    double* const d     = f.data();
    std::size_t const n = f.nsites();
    rng.normal_fill(d, n);
}

// ---- Lattice<std::complex<double>> ------------------------------------------
template <class Rng>
void hot_init(Lattice<std::complex<double>>& f, Rng& rng) noexcept {
    std::size_t const n = f.nsites();
    auto* const d       = f.data();
    // Layout guarantees Re/Im are adjacent doubles (C++ §29.5.4).
    rng.normal_fill(reinterpret_cast<double*>(d), 2 * n);
}

// ---- Lattice<std::array<double, N>> -----------------------------------------
template <std::size_t N, class Rng>
void hot_init(Lattice<std::array<double, N>>& f, Rng& rng) noexcept {
    std::size_t const ns = f.nsites();
    auto* const d        = f.data();
    for (std::size_t i = 0; i < ns; ++i) {
        double s = 0.0;
        for (std::size_t k = 0; k < N; ++k) {
            d[i][k] = rng.normal();
            s += d[i][k] * d[i][k];
        }
        // Normalise to the unit sphere — what OnSigma expects.
        double const inv = 1.0 / std::sqrt(s);
        for (std::size_t k = 0; k < N; ++k) {
            d[i][k] *= inv;
        }
    }
}

// ---- LinkLattice<double> ----------------------------------------------------
template <class Rng>
void hot_init(LinkLattice<double>& f, Rng& rng) noexcept {
    double* const d     = f.data();
    std::size_t const n = f.nlinks();
    rng.normal_fill(d, n);
}

// ---- MatrixLinkLattice<SU2, double> -----------------------------------------
// Initialise to identity, then drift each direction by a random algebra
// element (dt = 0.5) — gives a non-trivial SU(2) on every link.
template <class Rng>
void hot_init(MatrixLinkLattice<gauge_group::SU2, double>& f, Rng& rng) noexcept {
    std::size_t const d  = f.ndims();
    std::size_t const ns = f.nsites();
    // Zero buffer, write identity per link.
    std::size_t const total = d * gauge_group::SU2::n_real_components * ns;
    double* const data      = f.data();
    for (std::size_t i = 0; i < total; ++i) {
        data[i] = 0.0;
    }
    for (std::size_t mu = 0; mu < d; ++mu) {
        double* const blk = f.mu_block_data(mu);
        for (std::size_t s = 0; s < ns; ++s) {
            blk[(0 * ns) + s] = 1.0;
            blk[(6 * ns) + s] = 1.0;
        }
    }
    std::vector<double> scratch(gauge_group::SU2::n_real_components * ns);
    for (std::size_t mu = 0; mu < d; ++mu) {
        math::su2::sample_algebra_slab(scratch.data(), rng, ns);
        math::su2::expi_lmul_slab(f.mu_block_data(mu), scratch.data(), 0.5, ns);
    }
}

// ---- MatrixLinkLattice<SU3, double> -----------------------------------------
template <class Rng>
void hot_init(MatrixLinkLattice<gauge_group::SU3, double>& f, Rng& rng) noexcept {
    std::size_t const d  = f.ndims();
    std::size_t const ns = f.nsites();
    std::size_t const total = d * gauge_group::SU3::n_real_components * ns;
    double* const data      = f.data();
    for (std::size_t i = 0; i < total; ++i) {
        data[i] = 0.0;
    }
    for (std::size_t mu = 0; mu < d; ++mu) {
        double* const blk = f.mu_block_data(mu);
        for (std::size_t s = 0; s < ns; ++s) {
            blk[(0 * ns) + s]  = 1.0;  // Re U_{00}
            blk[(8 * ns) + s]  = 1.0;  // Re U_{11}
            blk[(16 * ns) + s] = 1.0;  // Re U_{22}
        }
    }
    std::vector<double> scratch(gauge_group::SU3::n_real_components * ns);
    for (std::size_t mu = 0; mu < d; ++mu) {
        math::su3::sample_algebra_slab(scratch.data(), rng, ns);
        math::su3::expi_lmul_slab(f.mu_block_data(mu), scratch.data(), 0.5, ns);
    }
}

// ---- MatrixLinkLattice<U1, double> ------------------------------------------
// U(1) "matrix" is just an angle; same as LinkLattice fill.
template <class Rng>
void hot_init(MatrixLinkLattice<gauge_group::U1, double>& f, Rng& rng) noexcept {
    double* const d     = f.data();
    std::size_t const n = f.ndims() * f.nsites();
    rng.normal_fill(d, n);
}

}  // namespace reticolo::bench
